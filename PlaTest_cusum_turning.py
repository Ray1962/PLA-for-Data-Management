import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np


def configure_plot_fonts():
    """設定可用的中文字型，避免圖表中文顯示成方塊。"""
    preferred_fonts = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "PMingLiU",
        "MingLiU",
        "SimHei",
        "Noto Sans CJK TC",
        "Arial Unicode MS",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    selected = [name for name in preferred_fonts if name in available]

    if selected:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    # 避免座標軸負號顯示成方塊
    plt.rcParams["axes.unicode_minus"] = False


def parse_spoes_csv(file_path):
    wavelength_map = {}
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        for line in f:
            text = line.strip()
            if text == "[Data]":
                break
            m = re.match(r"Wavelength(\d+)=([0-9.]+)", text)
            if m:
                idx = int(m.group(1))
                wavelength = float(m.group(2))
                wavelength_map[f"Ch1_{idx}"] = wavelength

        reader = csv.DictReader(f)
        rows = list(reader)

    return wavelength_map, rows


def pick_channel_by_wavelength(wavelength_map, target_wavelength):
    if not wavelength_map:
        raise ValueError("找不到任何波長設定。")
    best_channel = min(wavelength_map, key=lambda ch: abs(wavelength_map[ch] - target_wavelength))
    return best_channel, wavelength_map[best_channel]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def moving_average(y, window):
    if window <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


def mad(values):
    med = np.median(values)
    return np.median(np.abs(values - med))


def cusum_change_points(series, drift, threshold):
    """雙向 CUSUM，回傳可能的變化點 index。"""
    points = []
    gp = 0.0
    gn = 0.0

    for i, value in enumerate(series):
        gp = max(0.0, gp + value - drift)
        gn = max(0.0, gn - value - drift)

        if gp >= threshold or gn >= threshold:
            points.append(i)
            gp = 0.0
            gn = 0.0

    return np.array(points, dtype=int)


def refine_turning_points(y_smooth, change_points, search_radius=8):
    """在 CUSUM 點附近找 |二階差分| 最大位置，作為更準的轉折點。"""
    if len(change_points) == 0:
        return np.array([], dtype=int)

    dy = np.diff(y_smooth)
    ddy = np.diff(dy)
    refined = []

    for cp in change_points:
        center = int(cp)
        left = max(1, center - search_radius)
        right = min(len(y_smooth) - 2, center + search_radius)
        if right <= left:
            refined.append(center)
            continue

        seg_left = max(0, left - 1)
        seg_right = min(len(ddy) - 1, right - 1)
        if seg_right < seg_left:
            refined.append(center)
            continue

        local = np.abs(ddy[seg_left : seg_right + 1])
        best_rel = int(np.argmax(local))
        best = seg_left + best_rel + 1
        refined.append(best)

    return np.array(refined, dtype=int)


def merge_close_indices(indices, min_distance, max_index):
    if len(indices) == 0:
        return np.array([], dtype=int)

    idx = np.array(sorted(set(int(i) for i in indices if 0 <= i <= max_index)), dtype=int)
    if len(idx) == 0:
        return idx

    merged = [int(idx[0])]
    for value in idx[1:]:
        if value - merged[-1] < min_distance:
            merged[-1] = int((merged[-1] + value) // 2)
        else:
            merged.append(int(value))

    return np.array(merged, dtype=int)


def detect_turning_points_by_cusum(y, smooth_window=5, drift_k=0.5, threshold_k=8.0, min_distance=8):
    """使用 CUSUM 偵測轉折點，回傳索引。"""
    y_smooth = moving_average(y, smooth_window)
    dy = np.diff(y_smooth)
    if len(dy) < 3:
        return np.array([], dtype=int), y_smooth, 0.0, 0.0

    baseline = float(np.median(dy))
    centered = dy - baseline
    sigma = float(mad(centered) * 1.4826)
    if sigma <= 1e-12:
        sigma = float(np.std(centered))
    sigma = max(sigma, 1e-12)

    drift = drift_k * sigma
    threshold = threshold_k * sigma

    cp = cusum_change_points(centered, drift=drift, threshold=threshold)
    refined = refine_turning_points(y_smooth, cp, search_radius=max(4, smooth_window * 2))
    refined = merge_close_indices(refined, min_distance=min_distance, max_index=len(y) - 1)

    return refined, y_smooth, drift, threshold


def robust_slope_sigma(y_smooth):
    dy = np.diff(y_smooth)
    if len(dy) == 0:
        return dy, 0.0
    baseline = float(np.median(dy))
    centered = dy - baseline
    sigma = float(mad(centered) * 1.4826)
    if sigma <= 1e-12:
        sigma = float(np.std(centered))
    return centered, max(sigma, 1e-12)


def detect_steep_events(
    y_smooth,
    x_arr,
    y_arr,
    start_k=3.0,
    end_k=1.0,
    start_run=3,
    end_run=5,
    min_len=4,
    peak_quiet_run=3,
):
    """偵測陡升/陡降事件，輸出每段起點與終點。"""
    centered, sigma = robust_slope_sigma(y_smooth)
    if len(centered) == 0:
        return [], 0.0, 0.0

    start_th = start_k * sigma
    end_th = end_k * sigma
    events = []
    state = "idle"
    direction = None
    start_idx = 0
    extremum_idx = 0
    no_new_extremum_run = 0
    i = 0

    while i < len(centered):
        if state == "idle":
            if i + start_run <= len(centered) and np.all(centered[i : i + start_run] >= start_th):
                state = "active"
                direction = "up"
                start_idx = i
                extremum_idx = min(len(y_arr) - 1, start_idx)
                no_new_extremum_run = 0
                i += start_run
                continue
            if i + start_run <= len(centered) and np.all(centered[i : i + start_run] <= -start_th):
                state = "active"
                direction = "down"
                start_idx = i
                extremum_idx = min(len(y_arr) - 1, start_idx)
                no_new_extremum_run = 0
                i += start_run
                continue
            i += 1
            continue

        local_extremum_end = False
        cur_point_idx = min(len(y_arr) - 1, i + 1)
        if direction == "up":
            if y_arr[cur_point_idx] > y_arr[extremum_idx]:
                extremum_idx = cur_point_idx
                no_new_extremum_run = 0
            else:
                no_new_extremum_run += 1
            local_extremum_end = no_new_extremum_run >= peak_quiet_run
        else:
            if y_arr[cur_point_idx] < y_arr[extremum_idx]:
                extremum_idx = cur_point_idx
                no_new_extremum_run = 0
            else:
                no_new_extremum_run += 1
            local_extremum_end = no_new_extremum_run >= peak_quiet_run

        flat_end = i + end_run <= len(centered) and np.all(np.abs(centered[i : i + end_run]) <= end_th)
        reverse_end = False
        if i + start_run <= len(centered):
            if direction == "up":
                reverse_end = np.all(centered[i : i + start_run] <= -start_th)
            else:
                reverse_end = np.all(centered[i : i + start_run] >= start_th)

        if flat_end or reverse_end or local_extremum_end or i == len(centered) - 1:
            # 回溯找最後一個真正陡坡點，消除 end_run 前瞻造成的延遲
            last_steep = start_idx
            for j in range(start_idx, i):
                if abs(centered[j]) > end_th:
                    last_steep = j
            end_idx = min(len(y_arr) - 1, last_steep + 1)
            if local_extremum_end:
                if direction == "up":
                    end_idx = min(end_idx, extremum_idx)
                else:
                    end_idx = max(end_idx, extremum_idx)
            s_idx = max(0, start_idx)
            if end_idx - s_idx >= min_len:
                seg = centered[s_idx : min(i + 1, len(centered))]
                events.append(
                    {
                        "direction": direction,
                        "start_idx": int(s_idx),
                        "end_idx": int(end_idx),
                        "start_x": float(x_arr[s_idx]),
                        "end_x": float(x_arr[end_idx]),
                        "start_y": float(y_arr[s_idx]),
                        "end_y": float(y_arr[end_idx]),
                        "delta_y": float(y_arr[end_idx] - y_arr[s_idx]),
                        "max_abs_slope": float(np.max(np.abs(seg))) if len(seg) else 0.0,
                    }
                )
            state = "idle"
            direction = None
            no_new_extremum_run = 0
            i += 1
            continue

        i += 1

    return events, start_th, end_th


def reconstruction_rmse(x_arr, y_arr, keep_idx):
    """用線性內插重建，計算 RMSE。"""
    if len(keep_idx) < 2:
        return float("inf")
    x_keep = x_arr[keep_idx]
    y_keep = y_arr[keep_idx]
    y_hat = np.interp(x_arr, x_keep, y_keep)
    return float(np.sqrt(np.mean((y_arr - y_hat) ** 2)))


def sweep_cusum_parameters(y_arr, x_arr, target_keep_ratio=0.06):
    """自動掃描 CUSUM 參數，挑選最佳壓縮結果。"""
    smooth_candidates = [3, 5, 7]
    drift_candidates = [0.25, 0.5, 0.8, 1.1]
    threshold_candidates = [4.0, 6.0, 8.0, 10.0, 12.0]
    min_dist_candidates = [4, 8, 12, 16]

    y_span = float(np.max(y_arr) - np.min(y_arr))
    y_span = max(y_span, 1e-9)
    best = None

    for smooth_window in smooth_candidates:
        for drift_k in drift_candidates:
            for threshold_k in threshold_candidates:
                for min_distance in min_dist_candidates:
                    turning_idx, y_smooth, drift, threshold = detect_turning_points_by_cusum(
                        y_arr,
                        smooth_window=smooth_window,
                        drift_k=drift_k,
                        threshold_k=threshold_k,
                        min_distance=min_distance,
                    )

                    keep_idx = np.array(sorted(set([0, len(y_arr) - 1] + turning_idx.tolist())), dtype=int)
                    keep_ratio = len(keep_idx) / len(y_arr)
                    rmse = reconstruction_rmse(x_arr, y_arr, keep_idx)
                    rmse_norm = rmse / y_span

                    ratio_penalty = abs(keep_ratio - target_keep_ratio)
                    point_count_penalty = 0.0
                    if len(keep_idx) < 12:
                        point_count_penalty = 0.1

                    score = rmse_norm + 0.75 * ratio_penalty + point_count_penalty

                    candidate = {
                        "score": score,
                        "turning_idx": turning_idx,
                        "y_smooth": y_smooth,
                        "drift": drift,
                        "threshold": threshold,
                        "smooth_window": smooth_window,
                        "drift_k": drift_k,
                        "threshold_k": threshold_k,
                        "min_distance": min_distance,
                        "keep_idx": keep_idx,
                        "keep_ratio": keep_ratio,
                        "rmse": rmse,
                    }

                    if best is None or candidate["score"] < best["score"]:
                        best = candidate

    return best


PARAMS_FILE = Path(__file__).with_suffix(".json")


def load_params() -> dict:
    """從 JSON 參數檔載入上次使用的設定，檔案不存在時回傳空字典。"""
    if PARAMS_FILE.exists():
        try:
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_params(params: dict) -> None:
    """將目前使用的參數寫入 JSON 參數檔，供下次執行時作為預設值。"""
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


csv_path = r"G:\我的雲端硬碟\01_Working\林口美光\SPOES\20260306\0306-All.csv"

configure_plot_fonts()
params = load_params()
default_wavelength = params.get("wavelength", 703.8)

wavelength_map, rows = parse_spoes_csv(csv_path)

print("可用波長 (前 12 個):")
for ch, wl in list(wavelength_map.items())[:12]:
    print(f"  {ch}: {wl} nm")

user_input = input(f"請輸入目標波長 (nm)，直接 Enter 使用預設 {default_wavelength}: ").strip()
target_wavelength = default_wavelength if not user_input else float(user_input)

channel_col, matched_wavelength = pick_channel_by_wavelength(wavelength_map, target_wavelength)
print(f"\n選擇波長: {target_wavelength} nm -> 使用欄位 {channel_col} (實際 {matched_wavelength} nm)")

x_values = []
y_values = []
timestamps = []
for i, row in enumerate(rows):
    y = to_float(row.get(channel_col))
    if y is None:
        continue
    x = to_float(row.get("No"))
    if x is None:
        x = float(i)
    x_values.append(x)
    y_values.append(y)
    timestamps.append(row.get("Timestamp", ""))

if len(y_values) < 3:
    raise ValueError("有效資料點不足，無法進行壓縮。")

x_arr = np.array(x_values)
y_arr = np.array(y_values)
ts_arr = np.array(timestamps)
points = np.column_stack((x_arr, y_arr))

# 可輸入 x 軸顯示範圍
default_xmin = float(np.min(x_arr))
default_xmax = float(np.max(x_arr))
prompt_xmin = params.get("xmin", default_xmin)
prompt_xmax = params.get("xmax", default_xmax)
xmin_text = input(f"請輸入 x 軸最小值，直接 Enter 使用 {prompt_xmin}: ").strip()
xmax_text = input(f"請輸入 x 軸最大值，直接 Enter 使用 {prompt_xmax}: ").strip()
plot_xmin = float(prompt_xmin) if not xmin_text else float(xmin_text)
plot_xmax = float(prompt_xmax) if not xmax_text else float(xmax_text)
if plot_xmin >= plot_xmax:
    print("x 軸範圍輸入無效（最小值需小於最大值），改用完整資料範圍。")
    plot_xmin, plot_xmax = default_xmin, default_xmax

# CUSUM 參數
saved_mode = params.get("mode", "auto")
mode_text = input(f"參數模式：manual(手動)/auto(自動掃描)，預設 {saved_mode}: ").strip().lower()
if not mode_text:
    mode_text = saved_mode
use_auto = mode_text in ("auto", "a", "自動")

if use_auto:
    saved_target_ratio = params.get("target_keep_ratio", 0.06)
    target_ratio_text = input(f"自動掃描目標保留比例(0~1，預設 {saved_target_ratio}): ").strip()
    target_keep_ratio = saved_target_ratio if not target_ratio_text else float(target_ratio_text)
    target_keep_ratio = min(max(target_keep_ratio, 0.01), 0.5)

    best = sweep_cusum_parameters(y_arr, x_arr, target_keep_ratio=target_keep_ratio)
    turning_idx = best["turning_idx"]
    y_smooth = best["y_smooth"]
    drift = best["drift"]
    threshold = best["threshold"]
    smooth_window = best["smooth_window"]
    drift_k = best["drift_k"]
    threshold_k = best["threshold_k"]
    min_distance = best["min_distance"]
    keep_idx = best["keep_idx"]
    rmse = best["rmse"]

    print("\n[Auto Sweep] 已選最佳參數")
    print(
        f"smooth_window={smooth_window}, drift_k={drift_k}, "
        f"threshold_k={threshold_k}, min_distance={min_distance}"
    )
    print(f"score={best['score']:.6g}, keep_ratio={best['keep_ratio']*100:.2f}%, rmse={rmse:.6g}")
else:
    s_sw = params.get("smooth_window", 5)
    s_dk = params.get("drift_k", 0.5)
    s_tk = params.get("threshold_k", 8.0)
    s_md = params.get("min_distance", 8)

    smooth_window_text = input(f"平滑視窗 (預設 {s_sw}): ").strip()
    drift_k_text = input(f"CUSUM drift 係數 (預設 {s_dk}): ").strip()
    threshold_k_text = input(f"CUSUM threshold 係數 (預設 {s_tk}): ").strip()
    min_dist_text = input(f"轉折點最小間距 (預設 {s_md}): ").strip()

    smooth_window = s_sw if not smooth_window_text else max(1, int(smooth_window_text))
    drift_k = s_dk if not drift_k_text else float(drift_k_text)
    threshold_k = s_tk if not threshold_k_text else float(threshold_k_text)
    min_distance = s_md if not min_dist_text else max(1, int(min_dist_text))

    turning_idx, y_smooth, drift, threshold = detect_turning_points_by_cusum(
        y_arr,
        smooth_window=smooth_window,
        drift_k=drift_k,
        threshold_k=threshold_k,
        min_distance=min_distance,
    )
    keep_idx = np.array(sorted(set([0, len(points) - 1] + turning_idx.tolist())), dtype=int)

peak_quiet_default = int(params.get("peak_quiet_run", 3))
peak_quiet_text = input(
    f"陡變終點判定：連續幾點未再創新高/新低即結束 (預設 {peak_quiet_default}): "
).strip()
peak_quiet_run = peak_quiet_default if not peak_quiet_text else max(1, int(peak_quiet_text))

show_steep_markers_default = bool(params.get("show_steep_markers", True))
show_steep_markers_hint = "y" if show_steep_markers_default else "n"
show_steep_markers_text = input(
    f"是否在圖上顯示陡升/陡降三角形記號? (y/n，預設 {show_steep_markers_hint}): "
).strip().lower()
if not show_steep_markers_text:
    show_steep_markers = show_steep_markers_default
else:
    show_steep_markers = show_steep_markers_text in ("y", "yes", "1", "true", "是")

steep_events, steep_start_th, steep_end_th = detect_steep_events(
    y_smooth=y_smooth,
    x_arr=x_arr,
    y_arr=y_arr,
    start_k=3.0,
    end_k=1.0,
    start_run=3,
    end_run=5,
    min_len=4,
    peak_quiet_run=peak_quiet_run,
)

steep_idx = sorted(
    {
        idx
        for event in steep_events
        for idx in (event["start_idx"], event["end_idx"])
    }
)

final_keep_idx = np.array(sorted(set(keep_idx.tolist() + steep_idx)), dtype=int)
compressed = points[final_keep_idx]

visible_mask = (x_arr >= plot_xmin) & (x_arr <= plot_xmax)
if not np.any(visible_mask):
    visible_mask = np.ones_like(x_arr, dtype=bool)
visible_y = y_arr[visible_mask]
y_min = float(np.min(visible_y))
y_max = float(np.max(visible_y))
y_span = y_max - y_min
if y_span <= 1e-12:
    y_pad = max(abs(y_min) * 0.05, 1.0)
else:
    y_pad = y_span * 0.05
plot_ymin = y_min - y_pad
plot_ymax = y_max + y_pad

print(f"原始點數: {len(points)}")
print(f"CUSUM 轉折點數: {len(turning_idx)}")
print(f"Steep 事件數: {len(steep_events)} (start_th={steep_start_th:.6g}, end_th={steep_end_th:.6g})")
print(f"Steep 起訖點數: {len(steep_idx)}")
print(f"壓縮後保留點數: {len(compressed)}")
print(f"drift={drift:.6g}, threshold={threshold:.6g}")
print(
    f"使用參數: smooth_window={smooth_window}, drift_k={drift_k}, "
    f"threshold_k={threshold_k}, min_distance={min_distance}"
)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(points[:, 0], points[:, 1], color="steelblue", linewidth=0.8, label=f"原始資料 ({len(points)} 點)")
if len(turning_idx) > 0:
    axes[0].scatter(
        points[turning_idx, 0],
        points[turning_idx, 1],
        s=14,
        color="black",
        alpha=0.8,
        label=f"CUSUM 轉折點 ({len(turning_idx)} 點)",
    )
# 四類陡崩標記
rise_start_idx  = [e["start_idx"] for e in steep_events if e["direction"] == "up"]
rise_end_idx    = [e["end_idx"]   for e in steep_events if e["direction"] == "up"]
fall_start_idx  = [e["start_idx"] for e in steep_events if e["direction"] == "down"]
fall_end_idx    = [e["end_idx"]   for e in steep_events if e["direction"] == "down"]

marker_groups = [
    (rise_start_idx,  "^",  "limegreen", "陡升起點"),
    (rise_end_idx,    "v",  "green",     "陡升終點"),
    (fall_start_idx,  "v",  "tomato",    "陡降起點"),
    (fall_end_idx,    "^",  "red",       "陡降終點"),
]
if show_steep_markers:
    for idx_list, marker, color, label_text in marker_groups:
        if idx_list:
            axes[0].scatter(
                points[idx_list, 0],
                points[idx_list, 1],
                s=60, marker=marker, color=color, zorder=5,
                label=f"{label_text} ({len(idx_list)})",
            )
axes[0].set_ylabel("Value")
axes[0].set_title(f"CUSUM 轉折點偵測：{channel_col} ({matched_wavelength} nm)")
axes[0].set_xlim(plot_xmin, plot_xmax)
axes[0].set_ylim(plot_ymin, plot_ymax)
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    compressed[:, 0],
    compressed[:, 1],
    color="tomato",
    linewidth=1.2,
    marker="o",
    markersize=2,
    label=f"壓縮後保留點 ({len(compressed)} 點)",
)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Value")
axes[1].set_title("壓縮後：保留 CUSUM + Steep 起迄點")
axes[1].set_xlim(plot_xmin, plot_xmax)
axes[1].set_ylim(plot_ymin, plot_ymax)
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

ratio = len(compressed) / len(points) * 100
plt.suptitle(f"CUSUM 壓縮比較  壓縮率: {ratio:.1f}%", fontsize=13)
plt.tight_layout()
plt.show()

output_path = Path(__file__).with_name(f"compressed_cusum_{channel_col}_{matched_wavelength}nm.csv")
with open(output_path, "w", encoding="utf-8", newline="") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["x", channel_col, "is_cusum_turning", "is_steep_endpoint"])
    turning_set = set(turning_idx.tolist())
    steep_set = set(steep_idx)
    for idx in final_keep_idx:
        writer.writerow([points[idx, 0], points[idx, 1], int(idx in turning_set), int(idx in steep_set)])

print(f"縮減後資料已輸出: {output_path}")

# 陡崩事件時間軸 CSV
steep_timeline_path = Path(__file__).with_name(
    f"steep_timeline_{channel_col}_{matched_wavelength}nm.csv"
)
steep_rows = []
for event in steep_events:
    dir_zh = "陡升" if event["direction"] == "up" else "陡降"
    s_idx = event["start_idx"]
    e_idx = event["end_idx"]
    steep_rows.append({
        "x": event["start_x"],
        "timestamp": ts_arr[s_idx] if s_idx < len(ts_arr) else "",
        "value": event["start_y"],
        "direction": dir_zh,
        "event_type": f"{dir_zh}起點",
        "delta_y": event["delta_y"],
        "max_abs_slope": event["max_abs_slope"],
    })
    steep_rows.append({
        "x": event["end_x"],
        "timestamp": ts_arr[e_idx] if e_idx < len(ts_arr) else "",
        "value": event["end_y"],
        "direction": dir_zh,
        "event_type": f"{dir_zh}終點",
        "delta_y": event["delta_y"],
        "max_abs_slope": event["max_abs_slope"],
    })
steep_rows.sort(key=lambda r: r["x"])
with open(steep_timeline_path, "w", encoding="utf-8-sig", newline="") as sf:
    writer = csv.DictWriter(
        sf,
        fieldnames=["timestamp", "x", "value", "direction", "event_type", "delta_y", "max_abs_slope"],
    )
    writer.writeheader()
    writer.writerows(steep_rows)
print(f"陡崩時間軸已輸出: {steep_timeline_path} ({len(steep_rows)} 筆)")

save_params({
    "wavelength": target_wavelength,
    "xmin": plot_xmin,
    "xmax": plot_xmax,
    "mode": "auto" if use_auto else "manual",
    "target_keep_ratio": target_keep_ratio if use_auto else params.get("target_keep_ratio", 0.06),
    "smooth_window": smooth_window,
    "drift_k": drift_k,
    "threshold_k": threshold_k,
    "min_distance": min_distance,
    "peak_quiet_run": peak_quiet_run,
    "show_steep_markers": show_steep_markers,
})
print(f"參數已儲存: {PARAMS_FILE}")
