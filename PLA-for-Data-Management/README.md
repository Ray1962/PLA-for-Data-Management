# PLA - CUSUM Compression with Steep Events, Plateau Averages, and Change Rate

This project compresses SPOES time-series spectra while preserving key behavior:
- CUSUM turning points
- steep rise/fall start and end events
- plateau representative values computed by trapezoidal integral average
- plateau average percentage change rate with robust smoothing

Main script:
- PlaTest_cusum_turning.py

## Input Format

The script reads SPOES CSV files with:
- wavelength metadata in the header
- a [Data] section containing Timestamp, No, and Ch1_* columns

At runtime, you input a target wavelength (nm), and the script auto-selects the closest channel.

## Data Source Configuration

The CSV source path is currently set in the script as csv_path.
At runtime, the script prints:
- full source file path
- source filename

The source filename is also displayed in the figure title.

## Interactive Parameter Menu

Parameters are edited via a numbered menu and executed with 0.
Use Q to quit.

Current menu items:
1. x min
2. x max
3. mode (manual / auto)
4. target keep ratio (auto mode)
5. CUSUM smooth window
6. CUSUM drift k
7. CUSUM threshold k
8. CUSUM minimum turning-point distance
9. CUSUM refine search radius
10. steep end quiet-run length
11. steep detection smooth window
12. pre-peak percent gate (<0 means disabled)
13. show steep markers on subplot 1
14. plateau shrink percent

## Core Behavior

### CUSUM turning points
- Detects turning points from smoothed signal.
- Supports manual parameters or auto sweep mode.
- Refine stage is controlled by refine_search_radius.

### Steep event detection
- Detects rise/fall events with independent steep smoothing (steep_smooth_window).
- Uses relative gate pre_peak_percent based on previous local peak level.

### Plateau integral averages
- Pairs each rise end with the next fall start.
- Shrinks interval inward by plateau_shrink_pct.
- Computes representative average using trapezoid integral average:
	integral(y, x) / interval_width

### Percentage change rate (subplot 3)
- Builds plateau average series from plateau representative points.
- Smooths average values first.
- Computes percentage change rate from smoothed averages.
- Applies robust outlier suppression (MAD-based clipping).
- Applies moving-average smoothing to reduce jitter in the rate curve.

## Visualization (Single 3-Subplot Figure)

1. Original data + CUSUM turning points + optional steep markers
2. Compressed retained points
3. Plateau integral average and percentage change rate (dual y-axes)

Figure suptitle includes:
- channel and wavelength
- source filename

## Output Files

- compressed_cusum_<channel>_<wavelength>nm.csv
	- retained points and flags (is_cusum_turning, is_steep_endpoint)
- steep_timeline_<channel>_<wavelength>nm.csv
	- rise/fall timeline with timestamp, direction, event type, and slope metadata
- plateau_derivative_<channel>_<wavelength>nm.csv
	- avg_value rows first
	- percentage change rate rows after

## Parameter Persistence

Runtime settings are saved to:
- PlaTest_cusum_turning.json

## Quick Run

python PlaTest_cusum_turning.py
