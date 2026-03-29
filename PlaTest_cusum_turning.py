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


def detect_turning_points_by_cusum(y, smooth_window=5, drift_k=0.5, threshold_k=8.0, min_distance=8, refine_search_radius=None):
    """使用 CUSUM 偵測轉折點，回傳索引。

    refine_search_radius: refine 搜尋半徑，None 表示使用 max(4, smooth_window*2)。
    較小的值（如 2~4）可避免轉折點被拉離真實位置。
    """
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
    radius = refine_search_radius if refine_search_radius is not None else max(4, smooth_window * 2)
    refined = refine_turning_points(y_smooth, cp, search_radius=radius)
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


def find_previous_local_peak(y_values, ref_idx):
    """找 ref_idx 前方最近的局部高點，找不到則退回前段全域最高點。"""
    if len(y_values) < 3 or ref_idx <= 0:
        return None, None

    last_idx = min(int(ref_idx) - 1, len(y_values) - 2)
    for idx in range(last_idx, 0, -1):
        if y_values[idx] >= y_values[idx - 1] and y_values[idx] >= y_values[idx + 1]:
            return idx, float(y_values[idx])

    search_end = min(int(ref_idx), len(y_values))
    if search_end <= 0:
        return None, None
    peak_idx = int(np.argmax(y_values[:search_end]))
    return peak_idx, float(y_values[peak_idx])


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
    steep_smooth_window=5,
    pre_peak_percent=20.0,
):
    """偵測陡升/陡降事件，輸出每段起點與終點。

    steep_smooth_window：陡坡偵測專用平滑視窗，與 CUSUM 平滑視窗獨立。
    使用較小的視窗（預設 5）可保留真實斜率大小，避免大視窗壓低 sigma 導致誤判。
    pre_peak_percent：以前面局部高點為基準的百分比。
    陡升起點與陡降終點都必須低於「前面局部高點 * 百分比 / 100」。
    """
    # 用獨立視窗重新平滑，不依賴傳入的 CUSUM y_smooth
    y_steep = moving_average(y_arr, steep_smooth_window)
    centered, sigma = robust_slope_sigma(y_steep)
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
    last_down_gate_value = None
    i = 0

    while i < len(centered):
        if state == "idle":
            if i + start_run <= len(centered) and np.all(centered[i : i + start_run] >= start_th):
                state = "active"
                direction = "up"
                start_idx = i
                extremum_idx = min(len(y_steep) - 1, start_idx)
                no_new_extremum_run = 0
                i += start_run
                continue
            if i + start_run <= len(centered) and np.all(centered[i : i + start_run] <= -start_th):
                state = "active"
                direction = "down"
                start_idx = i
                extremum_idx = min(len(y_steep) - 1, start_idx)
                no_new_extremum_run = 0
                i += start_run
                continue
            i += 1
            continue

        local_extremum_end = False
        cur_point_idx = min(len(y_steep) - 1, i + 1)
        if direction == "up":
            if y_steep[cur_point_idx] > y_steep[extremum_idx]:
                extremum_idx = cur_point_idx
                no_new_extremum_run = 0
            else:
                no_new_extremum_run += 1
            local_extremum_end = no_new_extremum_run >= peak_quiet_run
        else:
            if y_steep[cur_point_idx] < y_steep[extremum_idx]:
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
                if pre_peak_percent is not None:
                    if direction == "up":
                        # 陡升起點直接沿用上一個陡降終點計算出的門檻值，不再重算。
                        if last_down_gate_value is not None and y_arr[s_idx] >= last_down_gate_value:
                            state = "idle"
                            direction = None
                            no_new_extremum_run = 0
                            i += 1
                            continue
                    else:
                        peak_ref_idx, peak_value = find_previous_local_peak(y_steep, s_idx)
                        if peak_value is not None:
                            gate_value = peak_value * float(pre_peak_percent) / 100.0
                            if y_arr[end_idx] >= gate_value:
                                state = "idle"
                                direction = None
                                no_new_extremum_run = 0
                                i += 1
                                continue
                            last_down_gate_value = gate_value
                seg = centered[s_idx : min(i + 1, len(centered))]
                events.append(
                    {
                        "direction": direction,
                        "start_idx": int(s_idx),
                        "end_idx": int(end_idx),
                        "start_x": float(x_arr[s_idx]),
                        "end_x": float(x_arr[end_idx]),
                        "start_y": float(y_arr[s_idx]),   # 輸出原始值（未平滑）
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

# ── 初始化所有參數 ──────────────────────────────────────────────────────────
default_xmin = float(np.min(x_arr))
default_xmax = float(np.max(x_arr))

_p = {
    "xmin":                float(params.get("xmin",                default_xmin)),
    "xmax":                float(params.get("xmax",                default_xmax)),
    "mode":                params.get("mode",                      "manual"),
    "target_keep_ratio":   float(params.get("target_keep_ratio",   0.06)),
    "smooth_window":       int(params.get("smooth_window",         5)),
    "drift_k":             float(params.get("drift_k",             0.5)),
    "threshold_k":         float(params.get("threshold_k",         8.0)),
    "min_distance":        int(params.get("min_distance",          8)),
    "refine_search_radius":int(params.get("refine_search_radius",  4)),
    "peak_quiet_run":      int(params.get("peak_quiet_run",        3)),
    "steep_smooth_window": int(params.get("steep_smooth_window",   5)),
    "pre_peak_percent_val":float(params.get("pre_peak_percent",    20.0)),
    "show_steep_markers":  bool(params.get("show_steep_markers",   True)),
    "plateau_shrink_pct":  float(params.get("plateau_shrink_pct",  5.0)),
}

def _show_menu(p):
    use_a = p["mode"] in ("auto", "a", "自動")
    mode_str = "auto（自動掃描）" if use_a else "manual（手動）"
    pre_str  = f"{p['pre_peak_percent_val']} %" if p["pre_peak_percent_val"] >= 0 else "不限制"
    sm_str   = "是" if p["show_steep_markers"] else "否"
    cusum_note = "  ← auto 模式自動決定" if use_a else ""
    print("\n" + "="*62)
    print("  目前參數設定")
    print("="*62)
    print(f"  ── 基本設定 ──")
    print(f"  [ 1]  x 軸最小值              : {p['xmin']}")
    print(f"  [ 2]  x 軸最大值              : {p['xmax']}")
    print(f"  [ 3]  參數模式                : {mode_str}")
    if use_a:
        print(f"  [ 4]  目標保留比例 (auto)     : {p['target_keep_ratio']}")
    print(f"\n  ── CUSUM 轉折點{cusum_note} ──")
    print(f"  [ 5]  平滑視窗                : {p['smooth_window']}")
    print(f"  [ 6]  drift 係數              : {p['drift_k']}")
    print(f"  [ 7]  threshold 係數          : {p['threshold_k']}")
    print(f"  [ 8]  轉折點最小間距          : {p['min_distance']}")
    print(f"  [ 9]  Refine 搜尋半徑         : {p['refine_search_radius']}")
    print(f"\n  ── 陡升/陡降事件 ──")
    print(f"  [10]  終點靜止判定點數        : {p['peak_quiet_run']}")
    print(f"  [11]  陡坡偵測平滑視窗        : {p['steep_smooth_window']}")
    print(f"  [12]  前峰百分比門檻          : {pre_str}  (負值=不限制)")
    print(f"  [13]  顯示事件標記            : {sm_str}")
    print(f"  [14]  平台區內縮比例 (%)      : {p['plateau_shrink_pct']}")
    print("="*62)
    print("  [ 0]  執行分析")
    print("="*62)

while True:
    _show_menu(_p)
    sel = input("請選擇要修改的參數編號，或按 0 執行: ").strip()
    if sel in ("0", ""):
        break
    elif sel == "1":
        v = input(f"  x 軸最小值 (目前 {_p['xmin']}): ").strip()
        if v: _p["xmin"] = float(v)
    elif sel == "2":
        v = input(f"  x 軸最大值 (目前 {_p['xmax']}): ").strip()
        if v: _p["xmax"] = float(v)
    elif sel == "3":
        v = input("  參數模式 (manual / auto): ").strip().lower()
        if v: _p["mode"] = v
    elif sel == "4":
        v = input(f"  目標保留比例 0~1 (目前 {_p['target_keep_ratio']}): ").strip()
        if v: _p["target_keep_ratio"] = min(max(float(v), 0.01), 0.5)
    elif sel == "5":
        v = input(f"  平滑視窗 (目前 {_p['smooth_window']}): ").strip()
        if v: _p["smooth_window"] = max(1, int(v))
    elif sel == "6":
        v = input(f"  drift 係數 (目前 {_p['drift_k']}): ").strip()
        if v: _p["drift_k"] = float(v)
    elif sel == "7":
        v = input(f"  threshold 係數 (目前 {_p['threshold_k']}): ").strip()
        if v: _p["threshold_k"] = float(v)
    elif sel == "8":
        v = input(f"  轉折點最小間距 (目前 {_p['min_distance']}): ").strip()
        if v: _p["min_distance"] = max(1, int(v))
    elif sel == "9":
        v = input(f"  Refine 搜尋半徑 (目前 {_p['refine_search_radius']}): ").strip()
        if v: _p["refine_search_radius"] = max(1, int(v))
    elif sel == "10":
        v = input(f"  終點靜止判定點數 (目前 {_p['peak_quiet_run']}): ").strip()
        if v: _p["peak_quiet_run"] = max(1, int(v))
    elif sel == "11":
        v = input(f"  陡坡偵測平滑視窗 (目前 {_p['steep_smooth_window']}): ").strip()
        if v: _p["steep_smooth_window"] = max(1, int(v))
    elif sel == "12":
        v = input(f"  前峰百分比門檻，負值=不限制 (目前 {_p['pre_peak_percent_val']}): ").strip()
        if v: _p["pre_peak_percent_val"] = float(v)
    elif sel == "13":
        v = input("  顯示事件標記 (y/n): ").strip().lower()
        if v: _p["show_steep_markers"] = v in ("y", "yes", "1", "true", "是")
    elif sel == "14":
        v = input(f"  平台區內縮比例 % (目前 {_p['plateau_shrink_pct']}): ").strip()
        if v: _p["plateau_shrink_pct"] = max(0.0, float(v))
    else:
        print("  無效的選擇，請重新輸入。")

# 解開參數
plot_xmin            = _p["xmin"]
plot_xmax            = _p["xmax"]
mode                 = _p["mode"]
target_keep_ratio    = _p["target_keep_ratio"]
smooth_window        = _p["smooth_window"]
drift_k              = _p["drift_k"]
threshold_k          = _p["threshold_k"]
min_distance         = _p["min_distance"]
refine_search_radius = _p["refine_search_radius"]
peak_quiet_run       = _p["peak_quiet_run"]
steep_smooth_window  = _p["steep_smooth_window"]
pre_peak_percent_val = _p["pre_peak_percent_val"]
pre_peak_percent     = None if pre_peak_percent_val < 0 else pre_peak_percent_val
show_steep_markers   = _p["show_steep_markers"]
plateau_shrink_pct   = _p["plateau_shrink_pct"]

if plot_xmin >= plot_xmax:
    print("x 軸範圍無效，改用完整資料範圍。")
    plot_xmin, plot_xmax = default_xmin, default_xmax

# ── 執行 CUSUM ──────────────────────────────────────────────────────────────
use_auto = mode in ("auto", "a", "自動")

if use_auto:
    best = sweep_cusum_parameters(y_arr, x_arr, target_keep_ratio=target_keep_ratio)
    turning_idx   = best["turning_idx"]
    y_smooth      = best["y_smooth"]
    drift         = best["drift"]
    threshold     = best["threshold"]
    smooth_window = best["smooth_window"]
    drift_k       = best["drift_k"]
    threshold_k   = best["threshold_k"]
    min_distance  = best["min_distance"]
    keep_idx      = best["keep_idx"]
    rmse          = best["rmse"]
    print("\n[Auto Sweep] 已選最佳參數")
    print(f"smooth_window={smooth_window}, drift_k={drift_k}, "
          f"threshold_k={threshold_k}, min_distance={min_distance}")
    print(f"score={best['score']:.6g}, keep_ratio={best['keep_ratio']*100:.2f}%, rmse={rmse:.6g}")
else:
    turning_idx, y_smooth, drift, threshold = detect_turning_points_by_cusum(
        y_arr,
        smooth_window=smooth_window,
        drift_k=drift_k,
        threshold_k=threshold_k,
        min_distance=min_distance,
        refine_search_radius=refine_search_radius,
    )
    keep_idx = np.array(sorted(set([0, len(points) - 1] + turning_idx.tolist())), dtype=int)

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
    steep_smooth_window=steep_smooth_window,
    pre_peak_percent=pre_peak_percent,
)

steep_idx = sorted(
    {
        idx
        for event in steep_events
        for idx in (event["start_idx"], event["end_idx"])
    }
)

# ── 陡升終點 → 陡降起點 平台區積分平均 ─────────────────────────────────────
def compute_plateau_segments(steep_events, x_arr, y_arr, shrink_pct):
    """找每個陡升終點之後最近的陡降起點，在內縮後的範圍計算梯形積分平均值。"""
    up_ends = sorted(
        [(e["end_idx"], e) for e in steep_events if e["direction"] == "up"],
        key=lambda t: t[0],
    )
    down_starts = sorted(
        [(e["start_idx"], e) for e in steep_events if e["direction"] == "down"],
        key=lambda t: t[0],
    )
    segments = []
    for rise_end_idx, _ in up_ends:
        # 找緊接在陡升終點之後的陡降起點
        match = next(
            ((fi, fe) for fi, fe in down_starts if fi > rise_end_idx), None
        )
        if match is None:
            continue
        fall_start_idx, _ = match
        x_a = float(x_arr[rise_end_idx])
        x_b = float(x_arr[fall_start_idx])
        width = x_b - x_a
        if width <= 0:
            continue
        margin = shrink_pct / 100.0 * width
        x_inner_a = x_a + margin
        x_inner_b = x_b - margin
        if x_inner_a >= x_inner_b:
            continue
        mask = (x_arr >= x_inner_a) & (x_arr <= x_inner_b)
        if np.sum(mask) < 2:
            continue
        x_seg = x_arr[mask]
        y_seg = y_arr[mask]
        integral = float(np.trapezoid(y_seg, x_seg))
        avg_value = integral / (x_inner_b - x_inner_a)
        segments.append({
            "rise_end_idx":   int(rise_end_idx),
            "fall_start_idx": int(fall_start_idx),
            "x_full_start":   x_a,
            "x_full_end":     x_b,
            "x_inner_start":  x_inner_a,
            "x_inner_end":    x_inner_b,
            "integral":       integral,
            "avg_value":      avg_value,
            "n_points":       int(np.sum(mask)),
        })
    return segments

plateau_segments = compute_plateau_segments(steep_events, x_arr, y_arr, plateau_shrink_pct)
if plateau_segments:
    print(f"\n平台區段數: {len(plateau_segments)}  (內縮比例 {plateau_shrink_pct}%)")
    for k, seg in enumerate(plateau_segments):
        print(
            f"  [{k+1}] x={seg['x_full_start']:.4g}~{seg['x_full_end']:.4g}  "
            f"內縮後 {seg['x_inner_start']:.4g}~{seg['x_inner_end']:.4g}  "
            f"avg={seg['avg_value']:.6g}  n={seg['n_points']}"
        )
else:
    print("\n未找到有效的陡升終點→陡降起點配對平台區段。")

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

# ── 建立重建序列（供第三圖使用）──────────────────────────────────────────
plateau_intervals = [
    (seg["x_full_start"], seg["x_full_end"],
     (seg["x_full_start"] + seg["x_full_end"]) / 2.0,
     seg["avg_value"])
    for seg in plateau_segments
] if plateau_segments else []

# 陡升/陡降區間 → 各取中間時間 + 平均值作為代表點
steep_rep_intervals = []
for e in steep_events:
    si, ei = e["start_idx"], e["end_idx"]
    xa, xb = float(x_arr[si]), float(x_arr[ei])
    seg_y = y_arr[si:ei + 1]
    avg_v = float(np.mean(seg_y)) if len(seg_y) > 0 else float(y_arr[si])
    steep_rep_intervals.append((xa, xb, (xa + xb) / 2.0, avg_v))

def in_any_steep(x_val):
    for xa, xb, _, _ in steep_rep_intervals:
        if xa <= x_val <= xb:
            return True
    return False

def in_any_plateau(x_val):
    for xa, xb, _, _ in plateau_intervals:
        if xa <= x_val <= xb:
            return True
    return False

rebuild_pts = []
for cx, cy in compressed:
    if in_any_steep(cx):
        pass  # 陡升陡降區域內壓縮點全部略去，改用代表點
    elif in_any_plateau(cx):
        is_plateau_endpoint = any(
            abs(cx - seg["x_full_start"]) < 1e-9 or abs(cx - seg["x_full_end"]) < 1e-9
            for seg in plateau_segments
        ) if plateau_segments else False
        if is_plateau_endpoint:
            rebuild_pts.append((cx, cy))
    else:
        rebuild_pts.append((cx, cy))

# 插入陡升/陡降代表點
for xa, xb, xmid, avg_v in steep_rep_intervals:
    rebuild_pts.append((xmid, avg_v))

for xa, xb, xmid, avg_val in plateau_intervals:
    rebuild_pts.append((xmid, avg_val))

rebuild_pts.sort(key=lambda t: t[0])
rebuild_x = np.array([p[0] for p in rebuild_pts])
rebuild_y = np.array([p[1] for p in rebuild_pts])

# ── 三張子圖合一 figure ───────────────────────────────────────────────────
ratio = len(compressed) / len(points) * 100
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 子圖一：原始資料 + CUSUM 轉折點 + 陡升陡降標記
rise_start_idx  = [e["start_idx"] for e in steep_events if e["direction"] == "up"]
rise_end_idx    = [e["end_idx"]   for e in steep_events if e["direction"] == "up"]
fall_start_idx  = [e["start_idx"] for e in steep_events if e["direction"] == "down"]
fall_end_idx    = [e["end_idx"]   for e in steep_events if e["direction"] == "down"]

axes[0].plot(points[:, 0], points[:, 1], color="steelblue", linewidth=0.8, label=f"原始資料 ({len(points)} 點)")
if len(turning_idx) > 0:
    axes[0].scatter(
        points[turning_idx, 0], points[turning_idx, 1],
        s=14, color="black", alpha=0.8,
        label=f"CUSUM 轉折點 ({len(turning_idx)} 點)",
    )
marker_groups = [
    (rise_start_idx, "^", "limegreen", "陡升起點"),
    (rise_end_idx,   "v", "green",     "陡升終點"),
    (fall_start_idx, "v", "tomato",    "陡降起點"),
    (fall_end_idx,   "^", "red",       "陡降終點"),
]
if show_steep_markers:
    for idx_list, marker, color, label_text in marker_groups:
        if idx_list:
            axes[0].scatter(
                points[idx_list, 0], points[idx_list, 1],
                s=60, marker=marker, color=color, zorder=5,
                label=f"{label_text} ({len(idx_list)})",
            )
axes[0].set_ylabel("Value")
axes[0].set_title(f"① CUSUM 轉折點偵測：{channel_col} ({matched_wavelength} nm)")
axes[0].set_ylim(plot_ymin, plot_ymax)
axes[0].legend(loc="upper right", fontsize=7, ncol=3)
axes[0].grid(True, alpha=0.3)

# 子圖二：壓縮後保留點
axes[1].plot(
    compressed[:, 0], compressed[:, 1],
    color="tomato", linewidth=1.2, marker="o", markersize=2,
    label=f"壓縮後保留點 ({len(compressed)} 點)",
)
axes[1].set_ylabel("Value")
axes[1].set_title(f"② 壓縮結果  壓縮率: {ratio:.1f}%")
axes[1].set_ylim(plot_ymin, plot_ymax)
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

# 子圖三：只畫平台積分平均值連線
plateau_rep_x = [xmid for _, _, xmid, _ in plateau_intervals]
plateau_rep_y = [avg  for _, _, _, avg  in plateau_intervals]

avg_pts = sorted([(x, y) for x, y in zip(plateau_rep_x, plateau_rep_y)], key=lambda t: t[0])
avg_x = np.array([p[0] for p in avg_pts])
avg_y = np.array([p[1] for p in avg_pts])

if len(avg_x) > 0:
    axes[2].plot(avg_x, avg_y, color="darkorange", linewidth=1.5,
                 label=f"積分平均值連線 ({len(avg_x)} 點)")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Value")
axes[2].set_title("③ 平台積分平均值連線")
axes[2].set_xlim(plot_xmin, plot_xmax)
axes[2].set_ylim(plot_ymin, plot_ymax)
axes[2].legend(loc="upper right", fontsize=7)
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"CUSUM 壓縮分析  {channel_col} ({matched_wavelength} nm)", fontsize=13)
fig.tight_layout()
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
    "refine_search_radius": refine_search_radius,
    "peak_quiet_run": peak_quiet_run,
    "steep_smooth_window": steep_smooth_window,
    "pre_peak_percent": pre_peak_percent_val,
    "plateau_shrink_pct": plateau_shrink_pct,
    "show_steep_markers": show_steep_markers,
})
print(f"參數已儲存: {PARAMS_FILE}")
