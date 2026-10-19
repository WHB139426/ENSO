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

Run the scripts in order:

```bash
python data.py
python preprocess.py --plots
python analysis.py
python model.py
python evaluate.py
```

The initial data download requires internet access. Use `--help` to see script options or `--overwrite` to replace existing outputs.

Main forecast outputs are `data/model/forecasts.nc` (SST fields and indices) and `data/model/forecast_indices.csv`. Evaluation results are saved under `data/evaluation/`.

## 4. Experimental results

The model uses six months of history to forecast three months ahead. Training covers 1850–1979, and validation covers 1980–1994. Held-out targets cover April 1995–December 2014, with 237 monthly forecasts after boundary exclusions. Validation selected 20 PCA components and a ridge penalty of `0.1`.

| Method | SST RMSE (°C) | Niño3.4 RMSE (°C) | Niño3.4 correlation |
| --- | ---: | ---: | ---: |
| Persistence | 0.7085 | 0.8085 | 0.8009 |
| PCA + ridge regression | 0.5154 | 0.5601 | 0.9154 |

Source: [data/evaluation/metrics.csv](data/evaluation/metrics.csv). These results evaluate CMIP6 simulation forecasts; the real-world forecast requires the instructor-provided hindcast data.
