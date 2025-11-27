# ENGR 010 Group Project - Electrical Engineering Option

## Overview
- Evaluates `power_system_data.csv` to highlight voltage, current, and real-power performance for substations SUB_001 to SUB_003 (Jan-Jun 2024).
- Outputs actionable metrics (basic statistics, standards compliance, power quality indices, fault summaries) and supporting visualisations.
- Provides an interactive dashboard for quick review alongside an automated reporting pipeline.

## Requirements
- Python 3.14 (tested with the project virtual environment).
- Recommended packages (install with `python -m pip install -r requirements.txt` if provided, or install individually):
	- `numpy`
	- `pandas`
	- `matplotlib`
	- `seaborn`
	- `pygame`
- Optional: `jupyter` for viewing `power_analysis_results.ipynb`.

## First-Time Setup
- (Optional) Create and activate a virtual environment: `python -m venv .venv` then `.venv\Scripts\activate` (PowerShell).
- Install dependencies listed above using `python -m pip install <package>`.
- Verify the CSV dataset `power_system_data.csv` is present in the repository root.

## Running the Batch Analysis
- Execute `python power_analysis_script.py` from the project root.
- The script reads `power_system_data.csv` via `analysis_core.py`, then:
	- Saves CSV summaries under `analysis_outputs/reports/`.
	- Stores trend charts and comparison plots under `analysis_outputs/figures/`.
	- Logs progress to the console for traceability.
- Rerun the script whenever the data file changes to refresh exported reports.

## Using the Dashboard Application
- Launch with `python main.py` (requires `pygame`).
- Behaviour:
	- Metrics in the left panel refresh directly from the CSV file; press `R` or click `Reload Data` to re-fetch.
	- Charts on the right rotate with `Prev`/`Next` buttons or arrow keys.
	- Switch between whole-dataset, monthly, and daily summaries with the `Cycle Mode` button (or press `M`).
	- Browse the available months/days using the `Prev/Next Period` buttons or the `[` and `]` keys.
	- Scroll the sidebar using the mouse wheel, Page Up/Down, Home/End, or the scrollbar to inspect all sections (including Fault Overview).
	- Exit using the `Exit` button, `Esc`, or `Q`.
- Resize support is fixed at 1280x720; adjust constants in `dashboard/app.py` if different dimensions are required.

## Data Updates and Maintenance
- Replace `power_system_data.csv` with a file using the same schema to analyse a new time window.
- Clear cached data (if the dashboard is running) with the `Reload Data` control; the analysis script reloads on each run.
- After modifying analytics logic inside `analysis_core.py`, rerun both the script and the dashboard to validate outputs.

## Troubleshooting
- If `pygame` fails to import, install it with `python -m pip install pygame`; Windows users may need Microsoft Visual C++ redistributables.
- Matplotlib back-end errors usually resolve by upgrading `matplotlib` or installing `pyqt` if interactive windows are desired for standalone plots.
- For performance issues during dashboard use, close other applications to free GPU/CPU resources, or lower the screen resolution constants.

## Change Log Snapshot
- **Nov 24, 2025**: Automated reporting pipeline added (`power_analysis_script.py`) with export directories and data science dependencies.
- **Nov 27, 2025**: Interactive Pygame dashboard introduced (`main.py`, `dashboard/`), sharing analytics via `analysis_core.py`; sidebar now scrollable for complete fault reporting.

