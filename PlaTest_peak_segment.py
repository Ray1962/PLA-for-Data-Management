import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def perpendicular_distance(p, line_start, line_end):
    line = line_end - line_start
    denom = np.linalg.norm(line)
    if denom == 0:
        return np.linalg.norm(p - line_start)
    return np.abs(line[0] * (line_start[1] - p[1]) - (line_start[0] - p[0]) * line[1]) / denom


def douglas_peucker_indices(points, indices, epsilon):
    """回傳被保留的原始 index（indices 是原始索引陣列）。"""
    if len(indices) < 3:
        return {int(indices[0]), int(indices[-1])}

    line_start = points[indices[0]]
    line_end = points[indices[-1]]

    dmax = -1.0
    split_pos = -1
    for pos in range(1, len(indices) - 1):
        idx = indices[pos]
        d = perpendicular_distance(points[idx], line_start, line_end)
        if d > dmax:
            dmax = d
            split_pos = pos

    if dmax > epsilon:
        left = douglas_peucker_indices(points, indices[: split_pos + 1], epsilon)
        right = douglas_peucker_indices(points, indices[split_pos:], epsilon)
        return left | right

    return {int(indices[0]), int(indices[-1])}


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


def detect_turning_points(y, prominence_ratio=0.05, window=6):
    """偵測局部峰/谷，並用近鄰 prominence 過濾雜訊。"""
    y = np.asarray(y)
    n = len(y)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    dy = np.diff(y)
    maxima = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    minima = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0] + 1

    span = float(np.max(y) - np.min(y))
    min_prom = span * prominence_ratio

    def keep_max(i):
        left = y[max(0, i - window) : i]
        right = y[i + 1 : min(n, i + window + 1)]
        if len(left) == 0 or len(right) == 0:
            return False
        prom = y[i] - max(np.min(left), np.min(right))
        return prom >= min_prom

    def keep_min(i):
        left = y[max(0, i - window) : i]
        right = y[i + 1 : min(n, i + window + 1)]
        if len(left) == 0 or len(right) == 0:
            return False
        prom = min(np.max(left), np.max(right)) - y[i]
        return prom >= min_prom

    maxima = np.array([i for i in maxima if keep_max(i)], dtype=int)
    minima = np.array([i for i in minima if keep_min(i)], dtype=int)
    return maxima, minima


def sample_range(start, end, min_samples, step_hint):
    if end < start:
        return []
    span = end - start + 1
    num = max(min_samples, span // max(step_hint, 1) + 1)
    return np.linspace(start, end, num=num, dtype=int).tolist()


def moving_average(y, window):
    if window <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


def has_run(values, start, run_length, comparator):
    end = start + run_length
    if start < 0 or end > len(values):
        return False
    return all(comparator(v) for v in values[start:end])


def detect_cusum_change_points(values, threshold, drift, direction):
    """在 1D 序列上用單向 CUSUM 偵測變化點。"""
    change_points = []
    accumulator = 0.0

    for idx, value in enumerate(values):
        if direction == "positive":
            accumulator = max(0.0, accumulator + value - drift)
        else:
            accumulator = max(0.0, accumulator - value - drift)

        if accumulator >= threshold:
            change_points.append(idx)
            accumulator = 0.0

    return np.array(change_points, dtype=int)


def pick_last_before(indices, pivot, default_value):
    indices = np.asarray(indices, dtype=int)
    candidates = indices[indices <= pivot]
    return int(candidates[-1]) if len(candidates) else int(default_value)


def pick_first_after(indices, pivot, default_value):
    indices = np.asarray(indices, dtype=int)
    candidates = indices[indices >= pivot]
    return int(candidates[0]) if len(candidates) else int(default_value)


def bridge_small_false_gaps(mask, max_gap):
    bridged = mask.copy()
    idx = 0
    n = len(mask)
    while idx < n:
        if bridged[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < n and not bridged[idx]:
            idx += 1
        gap_end = idx - 1
        gap_len = gap_end - gap_start + 1
        left_true = gap_start > 0 and bridged[gap_start - 1]
        right_true = idx < n and bridged[idx]
        if left_true and right_true and gap_len <= max_gap:
            bridged[gap_start : gap_end + 1] = True
    return bridged


def contiguous_true_region(mask, center_idx):
    if not mask[center_idx]:
        return center_idx, center_idx
    left = center_idx
    right = center_idx
    while left > 0 and mask[left - 1]:
        left -= 1
    while right + 1 < len(mask) and mask[right + 1]:
        right += 1
    return left, right


def detect_low_hold_region(y_smooth, min_idx, left_bound, right_bound, low_hold_level, min_width, gap_tolerance):
    local = y_smooth[left_bound : right_bound + 1]
    if len(local) == 0:
        return min_idx, min_idx
    low_mask = local <= low_hold_level
    low_mask = bridge_small_false_gaps(low_mask, max_gap=gap_tolerance)
    center = min_idx - left_bound
    region_left, region_right = contiguous_true_region(low_mask, center)

    if region_right - region_left + 1 < min_width:
        center_value = local[center]
        nearest = np.where(local <= center_value + (low_hold_level - center_value) * 0.6)[0]
        if len(nearest):
            region_left = max(0, int(np.min(nearest)))
            region_right = min(len(local) - 1, int(np.max(nearest)))

    return left_bound + region_left, left_bound + region_right


def detect_valley_features(y, minima_idx, maxima_idx, shoulder_ratio=0.75, floor_ratio=0.2):
    """用四狀態把 valley 分成高位平台、急降、低位平台、急升。"""
    y = np.asarray(y)
    n = len(y)
    maxima_idx = np.sort(np.asarray(maxima_idx, dtype=int))
    y_smooth = moving_average(y, window=5)
    dy = np.diff(y_smooth)
    features = []
    processed_pairs = set()

    for min_idx in np.sort(np.asarray(minima_idx, dtype=int)):
        left_candidates = maxima_idx[maxima_idx < min_idx]
        right_candidates = maxima_idx[maxima_idx > min_idx]

        left_peak = int(left_candidates[-1]) if len(left_candidates) else 0
        right_peak = int(right_candidates[0]) if len(right_candidates) else n - 1

        if left_peak >= min_idx or right_peak <= min_idx:
            continue

        pair_key = (left_peak, right_peak)
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        local_slice = slice(left_peak, right_peak + 1)
        local_min_rel = int(np.argmin(y_smooth[local_slice]))
        min_idx = left_peak + local_min_rel

        ref_level = min(y_smooth[left_peak], y_smooth[right_peak])
        depth = float(ref_level - y_smooth[min_idx])
        if depth <= 0:
            continue

        shoulder_level = y_smooth[min_idx] + depth * shoulder_ratio
        floor_level = y_smooth[min_idx] + depth * floor_ratio
        width = max(right_peak - left_peak, 1)
        local_slopes = np.abs(dy[left_peak:right_peak])
        slope_scale = float(np.percentile(local_slopes, 75)) if len(local_slopes) else 0.0
        edge_threshold = max(depth * 0.02, slope_scale * 1.8, 1e-9)
        flat_threshold = max(depth * 0.002, slope_scale * 0.25, 1e-9)
        edge_run = max(3, width // 40)
        floor_run = max(6, width // 18)

        local_dy = dy[left_peak:right_peak]
        cusum_threshold = edge_threshold * edge_run
        neg_change_rel = detect_cusum_change_points(
            local_dy,
            threshold=cusum_threshold,
            drift=flat_threshold,
            direction="negative",
        )
        pos_change_rel = detect_cusum_change_points(
            local_dy,
            threshold=cusum_threshold,
            drift=flat_threshold,
            direction="positive",
        )
        neg_change_idx = left_peak + neg_change_rel
        pos_change_idx = left_peak + pos_change_rel

        cusum_left_edge = pick_last_before(neg_change_idx, min_idx - 1, left_peak)
        cusum_right_edge = pick_first_after(pos_change_idx, min_idx, right_peak)

        left_shoulder = cusum_left_edge
        rise_start = cusum_right_edge
        low_hold_level = y_smooth[min_idx] + depth * min(floor_ratio, 0.08)
        left_floor, right_floor = detect_low_hold_region(
            y_smooth=y_smooth,
            min_idx=min_idx,
            left_bound=max(left_peak, cusum_left_edge),
            right_bound=min(right_peak, cusum_right_edge),
            low_hold_level=low_hold_level,
            min_width=max(5, floor_run // 2),
            gap_tolerance=max(1, floor_run // 4),
        )

        if right_floor - left_floor + 1 < max(4, floor_run // 3):
            left_floor = min_idx
            right_floor = min_idx
            rise_start = max(cusum_right_edge, min_idx)
        else:
            rise_start = max(cusum_right_edge, right_floor)

        for idx in range(left_floor, left_peak - 1, -1):
            if y_smooth[idx] >= shoulder_level:
                left_shoulder = idx
                break

        right_shoulder = right_peak
        for idx in range(rise_start, right_peak + 1):
            if y_smooth[idx] >= shoulder_level:
                right_shoulder = idx
                break

        left_floor = max(left_shoulder, min(left_floor, min_idx))
        right_floor = min(right_shoulder, max(right_floor, min_idx))

        ordered = [left_shoulder, left_floor, int(min_idx), right_floor, right_shoulder]
        ordered = [max(0, min(n - 1, idx)) for idx in ordered]

        if not (ordered[0] <= ordered[1] <= ordered[2] <= ordered[3] <= ordered[4]):
            continue

        valley_span = ordered[4] - ordered[0] + 1
        floor_span = ordered[3] - ordered[1] + 1

        descent_idx = sample_range(ordered[0], ordered[1], min_samples=4, step_hint=5)
        floor_idx = sample_range(ordered[1], ordered[3], min_samples=7, step_hint=3)
        ascent_idx = sample_range(ordered[3], ordered[4], min_samples=4, step_hint=5)

        support_idx = set(descent_idx)
        support_idx.update(floor_idx)
        support_idx.update(ascent_idx)
        support_idx.update(ordered)

        features.append(
            {
                "left_shoulder": ordered[0],
                "left_floor": ordered[1],
                "min_idx": ordered[2],
                "right_floor": ordered[3],
                "right_shoulder": ordered[4],
                "cusum": {
                    "fall_start": int(cusum_left_edge),
                    "rise_start": int(cusum_right_edge),
                },
                "states": {
                    "high_hold": (left_peak, ordered[0]),
                    "falling_edge": (ordered[0], ordered[1]),
                    "low_hold": (ordered[1], ordered[3]),
                    "rising_edge": (ordered[3], ordered[4]),
                },
                "support_idx": sorted(support_idx),
            }
        )

    return features


def estimate_cycle_segments(y, maxima_idx):
    """根據主要峰值估計週期邊界，回傳 [(start, end), ...]。"""
    n = len(y)
    if n < 2:
        return [(0, n - 1)]

    if len(maxima_idx) < 2:
        return [(0, n - 1)]

    peak_values = y[maxima_idx]
    cutoff = np.percentile(peak_values, 70)
    major_peaks = np.sort(maxima_idx[peak_values >= cutoff])

    if len(major_peaks) < 2:
        major_peaks = np.sort(maxima_idx)

    if len(major_peaks) < 2:
        return [(0, n - 1)]

    mids = [int((major_peaks[i] + major_peaks[i + 1]) // 2) for i in range(len(major_peaks) - 1)]
    boundaries = [0] + mids + [n - 1]

    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        segments.append((start, end))

    if not segments:
        segments = [(0, n - 1)]

    return segments


def interval_valley_factor(a, b, valley_features, valley_factor):
    for valley in valley_features:
        floor_start, floor_end = valley["states"]["low_hold"]
        descent_start, descent_end = valley["states"]["falling_edge"]
        ascent_start, ascent_end = valley["states"]["rising_edge"]
        if a <= floor_end and b >= floor_start:
            return valley_factor * 0.45
        if (a <= descent_end and b >= descent_start) or (a <= ascent_end and b >= ascent_start):
            return valley_factor
    return 1.0


def cycle_aware_compress(points, locked_idx, segments, valley_features, epsilon_ratio=0.01, valley_factor=0.35):
    """週期分段 + 鎖定點保留壓縮。"""
    y = points[:, 1]
    selected = set()

    for seg_start, seg_end in segments:
        seg_span = float(np.max(y[seg_start : seg_end + 1]) - np.min(y[seg_start : seg_end + 1]))
        seg_epsilon = max(seg_span * epsilon_ratio, 1e-9)

        anchors = [seg_start, seg_end]
        anchors.extend([idx for idx in locked_idx if seg_start <= idx <= seg_end])
        anchors = sorted(set(anchors))

        selected.update(anchors)

        for i in range(len(anchors) - 1):
            a = anchors[i]
            b = anchors[i + 1]
            if b - a < 2:
                selected.add(a)
                selected.add(b)
                continue
            sub_indices = np.arange(a, b + 1)
            local_factor = interval_valley_factor(a, b, valley_features, valley_factor)
            selected.update(douglas_peucker_indices(points, sub_indices, seg_epsilon * local_factor))

    selected.add(0)
    selected.add(len(points) - 1)
    keep = np.array(sorted(selected), dtype=int)
    return keep, points[keep]


csv_path = r"G:\我的雲端硬碟\01_Working\林口美光\SPOES\20260306\0306-All.csv"
default_wavelength = 703.8

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
for i, row in enumerate(rows):
    y = to_float(row.get(channel_col))
    if y is None:
        continue
    x = to_float(row.get("No"))
    if x is None:
        x = float(i)
    x_values.append(x)
    y_values.append(y)

if len(y_values) < 3:
    raise ValueError("有效資料點不足，無法進行壓縮。")

points = np.column_stack((np.array(x_values), np.array(y_values)))

# 可輸入 x 軸顯示範圍，留空代表使用完整範圍
default_xmin = float(np.min(points[:, 0]))
default_xmax = float(np.max(points[:, 0]))
xmin_text = input(f"請輸入 x 軸最小值，直接 Enter 使用 {default_xmin}: ").strip()
xmax_text = input(f"請輸入 x 軸最大值，直接 Enter 使用 {default_xmax}: ").strip()

plot_xmin = default_xmin if not xmin_text else float(xmin_text)
plot_xmax = default_xmax if not xmax_text else float(xmax_text)

if plot_xmin >= plot_xmax:
    print("x 軸範圍輸入無效（最小值需小於最大值），改用完整資料範圍。")
    plot_xmin, plot_xmax = default_xmin, default_xmax

maxima_idx, minima_idx = detect_turning_points(points[:, 1], prominence_ratio=0.05, window=6)
valley_features = detect_valley_features(points[:, 1], minima_idx, maxima_idx)
valley_lock_idx = np.array(
    sorted(
        {
            idx
            for valley in valley_features
            for idx in valley["support_idx"]
        }
    ),
    dtype=int,
)
locked_idx = np.sort(np.unique(np.concatenate([maxima_idx, minima_idx, valley_lock_idx])))
segments = estimate_cycle_segments(points[:, 1], maxima_idx)

keep_idx, compressed = cycle_aware_compress(
    points=points,
    locked_idx=locked_idx,
    segments=segments,
    valley_features=valley_features,
    epsilon_ratio=0.008,
    valley_factor=0.12,
)

print(f"原始點數: {len(points)}, 壓縮後點數: {len(compressed)}")
print(f"鎖定點數量(峰+谷): {len(locked_idx)}")
print(f"偵測週期段數: {len(segments)}")
print(f"偵測 valley 數量: {len(valley_features)}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(points[:, 0], points[:, 1], color="steelblue", linewidth=0.8, label=f"原始資料 ({len(points)} 點)")
if len(locked_idx) > 0:
    axes[0].scatter(
        points[locked_idx, 0],
        points[locked_idx, 1],
        s=10,
        color="black",
        alpha=0.6,
        label=f"鎖定峰谷 ({len(locked_idx)} 點)",
    )
if len(valley_lock_idx) > 0:
    axes[0].scatter(
        points[valley_lock_idx, 0],
        points[valley_lock_idx, 1],
        s=14,
        color="limegreen",
        alpha=0.85,
        label=f"Valley 特徵點 ({len(valley_lock_idx)} 點)",
    )
axes[0].set_ylabel("Value")
axes[0].set_title(f"壓縮前：{channel_col} ({matched_wavelength} nm)")
axes[0].set_xlim(plot_xmin, plot_xmax)
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    compressed[:, 0],
    compressed[:, 1],
    color="tomato",
    linewidth=1.1,
    marker="o",
    markersize=2,
    label=f"峰值鎖定+分段壓縮 ({len(compressed)} 點)",
)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Value")
axes[1].set_title("壓縮後：Peak-Locked + Cycle-Aware")
axes[1].set_xlim(plot_xmin, plot_xmax)
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

for seg_start, seg_end in segments:
    xline = points[seg_start, 0]
    if plot_xmin <= xline <= plot_xmax:
        axes[0].axvline(xline, color="gray", linestyle="--", alpha=0.15, linewidth=0.8)
        axes[1].axvline(xline, color="gray", linestyle="--", alpha=0.15, linewidth=0.8)

ratio = len(compressed) / len(points) * 100
plt.suptitle(f"Peak-Locked + Cycle-Aware 壓縮比較  壓縮率: {ratio:.1f}%", fontsize=13)
plt.tight_layout()
plt.show()

output_path = Path(__file__).with_name(f"compressed_peakcycle_{channel_col}_{matched_wavelength}nm.csv")
with open(output_path, "w", encoding="utf-8", newline="") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["x", channel_col])
    writer.writerows(compressed)

print(f"縮減後資料已輸出: {output_path}")
