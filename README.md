# PLA - CUSUM Turning Points and Steep Event Compression

This project compresses SPOES time-series spectra and preserves key behavior:
- CUSUM turning points
- steep rise/fall start and end events

Main script:
- `PlaTest_cusum_turning.py`

## Data and Output

Input data format (SPOES CSV):
- header region with wavelength metadata
- `[Data]` section with `Timestamp`, `No`, and `Ch1_*` channels

Main outputs:
- `compressed_cusum_<channel>_<wavelength>nm.csv`
- `steep_timeline_<channel>_<wavelength>nm.csv`

## Recent Update Process

### 1) Independent steep detection smoothing
Commit: `cfd66f0`

What changed:
- Added `steep_smooth_window` independent from CUSUM `smooth_window`.
- Steep event slope/sigma detection now uses this dedicated smoothing path.

Why:
- Avoid steep-event misdetection when CUSUM smoothing is large.

### 2) CUSUM refinement radius configurable + grouped prompts
Commit: `ea41be1`

What changed:
- Added `refine_search_radius` for turning-point refine stage.
- Input flow now visually separated into two blocks:
  - CUSUM parameters
  - steep rise/fall parameters

Why:
- Better alignment to real turning locations.
- Better input UX and clearer parameter grouping.

### 3) Strength gate changed to pre-peak percentage
Commit: `3789f5d`

What changed:
- Replaced fixed absolute thresholds with `pre_peak_percent`.
- Gate rule:
  - steep rise start and steep fall end must be below
    `previous local peak * pre_peak_percent / 100`.
- For rise-start validation, reuse the gate value computed in previous fall-end step (no recalculation).

Why:
- Relative threshold is more robust across different intensity ranges.
- Reusing previous fall gate keeps rise/fall pairing behavior consistent.

## Current Key Parameters

CUSUM related:
- `smooth_window`
- `drift_k`
- `threshold_k`
- `min_distance`
- `refine_search_radius`

Steep event related:
- `peak_quiet_run`
- `steep_smooth_window`
- `pre_peak_percent`
- `show_steep_markers`

## Parameter Persistence

Run-time inputs are saved in:
- `PlaTest_cusum_turning.json`

## Quick Run

```bash
python PlaTest_cusum_turning.py
```
