<<<<<<< HEAD
# Renal Glomeruli Analysis

This project computes lumen-based glomeruli metrics from `.svs` + `.svs.json` annotations, writes Python overlays, exports QuPath CSVs, performs artery flagging + severity grading, generates case summaries (JSON + PDF), and optionally runs QuPath headless.

## Prerequisites

- Windows or Linux
- Python 3.10+
- QuPath 0.6.x (only needed if you want QuPath plotting)

## Setup

From `Renal-Glomeruli-Analysis`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or use Conda:

```bash
conda env create -f environment.windows.yml
# or
conda env create -f environment.linux.yml
conda activate glomeruli
```

## Run

### 1) Default run (auto-picks dataset pair)

```powershell
python run_pipeline.py
```

### 2) Run a specific image/json pair

```powershell
python run_pipeline.py --svs ".\dataset\grundium17.svs" --json ".\dataset\grundium17.svs.json"
```

You can also pass absolute paths:

```powershell
python run_pipeline.py --svs "C:\path\my_slide.svs" --json "C:\path\my_slide.svs.json"
```

Rules for `--svs` and `--json`:

- Provide both arguments together.
- JSON filename must match the SVS stem.
- Example: `abc.svs` requires `abc.svs.json`.

### 3) Run without QuPath

```powershell
python run_pipeline.py --svs ".\dataset\grundium17.svs" --json ".\dataset\grundium17.svs.json" --no-qupath
```

### 4) Optional custom output paths for new summary artifacts

```powershell
python run_pipeline.py `
  --svs ".\dataset\grundium17.svs" `
  --json ".\dataset\grundium17.svs.json" `
  --output-flags-csv ".\output\grundium17\grundium17_artery_flags.csv" `
  --output-severity-csv ".\output\grundium17\grundium17_severity.csv" `
  --output-case-summary-json ".\output\grundium17\grundium17_case_summary.json" `
  --output-case-summary-pdf ".\output\grundium17\grundium17_case_summary.pdf"
```

## Output layout

For an input `grundium16.svs`:

- Python overlays + analysis JSON:
- Python overlays + analysis JSON:
  - `output\grundium16\grundium16_artery_overlay.png`
  - `output\grundium16\grundium16_aif_overlay.png`
  - `output\grundium16\grundium16_artery_analysis.json`
  - `output\grundium16\grundium16_artery_flags.csv`
  - `output\grundium16\grundium16_severity.csv`
  - `output\grundium16\grundium16_case_summary.json`
  - `output\grundium16\grundium16_case_summary.pdf`
- QuPath per-image project (auto-created):
  - `qupath_projects\grundium16\project.qpproj`
  - `qupath_projects\grundium16\grundium16.svs` (copied slide)
  - `qupath_projects\grundium16\output\*.csv`

This prevents one image run from overwriting another image run.

## New medical support outputs

- `*_artery_flags.csv`:
  - artery-level review priority (`urgent_review`, `review_recommended`, `routine`)
  - risk score and machine-generated reasons
  - includes advanced fields: `concentricity_index`, `wall_thickness_cv`, `narrowing_index`, `lumen_count`, `multi_lumen_complexity`, `contour_quality_mean`
- `*_severity.csv`:
  - artery-level severity grade (`0` to `3`) + severity label
- `*_case_summary.json`:
  - case-level totals, homogeneity/flag/severity distributions, impression
  - advanced cohort-ready metrics (`mean_aif_multi_percent`, `mean_concentricity_index`, `mean_wall_thickness_cv`, `mean_narrowing_index`, `multi_lumen_arteries`)
- `*_case_summary.pdf`:
  - formatted one-page deterministic summary report for review
  - includes advanced morphology panel + coherent top-flag table tags (CI/ML)
- QuPath advanced overlay CSVs (`qupath_projects\<case>\output\`):
  - `artery_metrics.csv` (CI, multi-lumen, wall-thickness variability, narrowing, risk/grade)
  - `ray_thickness_segments.csv` (ray-wise wall-thickness segments for direct QuPath rendering)

## Advanced morphology methods included

- Multi-lumen aware AIF:
  - uses lumen union area per artery instead of only the largest lumen contour
- Ray-casting wall profile:
  - computes per-angle wall thickness and narrowing index from artery centroid rays
- Concentricity index:
  - normalized off-center metric replacing plain centroid Euclidean distance
- Adaptive per-artery lumen segmentation:
  - combines Otsu + adaptive thresholding inside artery ROI for robust contour extraction

Note: These outputs are decision-support artifacts for research workflows and are not standalone clinical diagnoses.

## Git hygiene

Generated artifacts are intentionally ignored via `.gitignore`:

- `output/`
- `qupath_projects/`
- `__pycache__/`

## QuPath executable

The script auto-detects QuPath from `%LOCALAPPDATA%\QuPath-*`.

If auto-detection fails, set:

```powershell
setx QUPATH_EXE "C:\Users\<YOUR_USER>\AppData\Local\QuPath-0.6.0\QuPath-0.6.0 (console).exe"
```

Then open a new terminal and rerun.

## Troubleshooting

- `SVS file not found`:
  - Check the path passed to `--svs`.
- `JSON file not found`:
  - Check the path passed to `--json`.
- `Annotation JSON does not match the SVS file`:
  - Rename JSON to `<svs_stem>.svs.json` or use the matching file.
- `Could not find QuPath executable`:
  - Set `QUPATH_EXE` as shown above.
=======
# Renal-Glomeruli-Analysis
Python-based renal image analysis system that analyzes annotated kidney tissue images, extracts glomerular features, assesses vessel lumen narrowing, and generates CSV results and visual overlays to support renal artery blockage assessment.
>>>>>>> 99d5db0ad7cccb72e341b33f6545b76d28f82284
