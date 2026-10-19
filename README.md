# ENSO Forecasting: IPSL-CM6A-LR

Data processing, ENSO analysis, and three-month SST forecasting for ECS289G.

## 1. Environment setup

```bash
cd /path/to/project
conda create -n enso-project python=3.13 pip -y
conda activate enso-project
python -m pip install -r requirements.txt
```

Replace `/path/to/project` with your project directory. Activate `enso-project` whenever you open a new terminal.

## 2. Project structure

```text
.
├── ECS289G_ENSO_project.ipynb  # Reference notebook
├── data.py                   # Download and cache CMIP6 SST
├── preprocess.py             # Calculate anomalies and Niño indices
├── analysis.py               # Characterize simulated ENSO
├── model.py                  # Train persistence and PCA + ridge models
├── evaluate.py               # Evaluate forecasts and generate figures
├── requirements.txt          # Python dependencies
└── data/                     # Generated data and results
    ├── raw/                  # Cached SST and tropical background
    ├── processed/            # Climatologies, anomalies, and indices
    ├── analysis/             # ENSO statistics and composite maps
    ├── model/                # Saved model, forecast SST, and indices
    └── evaluation/           # Metrics and evaluation figures
```

## 3. Run

Run the following steps in order. All generated files are saved under `data/`.

**Step 1: Download the data**

```bash
python data.py
```

Downloads monthly historical SST from IPSL-CM6A-LR, extracts the tropical Pacific, and retains grid-cell areas and the tropical mean SST needed for relative anomalies.

Output: `data/raw/IPSL-CM6A-LR_historical.nc`, the local cache used by later scripts. This step requires internet access.

**Step 2: Preprocess SST and calculate indices**

```bash
python preprocess.py --plots
```

Calculates monthly climatology, SST anomalies, relative anomalies with the tropical background removed, Niño indices, and three-month index averages. The `--plots` option generates diagnostic figures.

Outputs in `data/processed/`: `pacific_ssta.nc` (anomaly fields), `monthly_climatology.nc`, `enso_indices.nc`, `enso_indices.csv`, and SST/index plots in `figures/`.

**Step 3: Analyze simulated ENSO**

```bash
python analysis.py
```

Summarizes ENSO variability, identifies warm and cold events, estimates event durations and dominant periods, and computes autocorrelations and event composite maps.

Outputs in `data/analysis/`: statistics and event tables, `spectrum.csv`, `autocorrelation.csv`, `enso_composites.nc`, and four summary plots in `figures/`.

**Step 4: Train the forecasting model**

```bash
python model.py
```

Uses six months of SST history to forecast three months ahead with PCA and ridge regression. It selects model settings on the validation set, generates validation and test forecasts, and includes a persistence baseline. Niño3.4 indices are calculated from the forecast SST fields.

Outputs in `data/model/`: `model.joblib` (fitted model), `forecasts.nc` (forecast SST fields, reference fields, and indices), `forecast_indices.csv`, `training_climatology.nc`, and tables recording validation scores and sample splits.

**Step 5: Evaluate the forecasts**

```bash
python evaluate.py
```

Checks forecast lead times, data splits, and SST-derived indices, then compares the model with persistence using SST errors, Niño3.4 errors and correlations, and forecast skill.

Outputs in `data/evaluation/`: `metrics.csv` (main results), `monthly_errors.csv`, `spatial_errors.nc`, and four comparison plots in `figures/`.

Use `--help` with any script to see its options. Add `--overwrite` when rerunning a step to replace existing outputs.

## 4. Experimental results

The model uses six months of history to forecast three months ahead. Training covers 1850–1979, and validation covers 1980–1994. Held-out targets cover April 1995–December 2014, with 237 monthly forecasts after boundary exclusions. Validation selected 20 PCA components and a ridge penalty of `0.1`.

| Method | SST RMSE (°C) | Niño3.4 RMSE (°C) | Niño3.4 correlation |
| --- | ---: | ---: | ---: |
| Persistence | 0.7085 | 0.8085 | 0.8009 |
| PCA + ridge regression | 0.5154 | 0.5601 | 0.9154 |

Source: [data/evaluation/metrics.csv](data/evaluation/metrics.csv). These results evaluate CMIP6 simulation forecasts; the real-world forecast requires the instructor-provided hindcast data.
