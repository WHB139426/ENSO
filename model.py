"""Train a three-month SST forecaster and a full-field persistence baseline.

Run: python model.py
Preprocessing is fitted on training years; tuning uses validation SST RMSE only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import logging
from pathlib import Path
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from data import ROOT, MODEL, Region, month_number, select_months, validate_monthly_time, write_netcdf
from preprocess import fit_climatology, validate_cache

LOG = logging.getLogger(__name__)
METHODS = ("persistence", "pca_ridge")
OUTPUTS = ("model.joblib", "training_climatology.nc", "forecasts.nc", "forecast_indices.csv",
           "validation_scores.csv", "sample_splits.csv", "training_metadata.json")


@dataclass(frozen=True)
class ForecastConfig:
    train_start: str = "1850-01"
    train_end: str = "1979-12"
    validation_end: str = "1994-12"
    test_end: str = "2014-12"
    base_start: str | None = None
    base_end: str | None = None
    lookback: int = 6
    lead: int = 3
    components: tuple[int, ...] = (10, 20, 40, 80)
    alphas: tuple[float, ...] = (0.1, 1., 10., 100.)
    seed: int = 42
    threads: int = 4

    def validate(self):
        start, train, valid, end = map(month_number, (self.train_start, self.train_end,
                                                      self.validation_end, self.test_end))
        if not start <= train < valid < end:
            raise ValueError("Require train_start <= train_end < validation_end < test_end.")
        if self.lookback < 1 or self.lead < 1 or self.threads < 1:
            raise ValueError("Lookback, lead and threads must be positive.")
        if not self.components or any(k < 1 for k in self.components):
            raise ValueError("PCA component counts must be positive.")
        if not self.alphas or any(not np.isfinite(a) or a <= 0 for a in self.alphas):
            raise ValueError("Ridge alphas must be finite and positive.")


def month_string(code: int) -> str:
    return f"{code//12:04d}-{code%12+1:02d}"


def labels(time):
    return [f"{int(y):04d}-{int(m):02d}" for y,m in zip(time.dt.year.values, time.dt.month.values)]


def sample_splits(time: xr.DataArray, config: ForecastConfig):
    """Purge boundary samples whose initialization predates the previous cutoff."""
    codes = validate_monthly_time(time)
    origins = np.arange(config.lookback-1, len(codes)-config.lead)
    targets = origins + config.lead
    split = np.full(len(origins), "purged", dtype="U10")
    train, valid = month_number(config.train_end), month_number(config.validation_end)
    split[codes[targets] <= train] = "train"
    split[(codes[origins] > train) & (codes[targets] <= valid)] = "validation"
    split[(codes[origins] > valid) & (codes[targets] <= month_number(config.test_end))] = "test"
    for name in ("train", "validation", "test"):
        if (split == name).sum() < 2:
            raise ValueError(f"Not enough {name} samples after forecast-boundary purging.")
    dates = np.array(labels(time))
    table = pd.DataFrame({"history_start_month": dates[origins-config.lookback+1],
                          "initialization_month": dates[origins], "target_month": dates[targets], "split": split})
    return origins, targets, split, table


def relative_vectors(cache, climate, mask):
    """Apply frozen climatology, without fitting anything on the supplied data."""
    months = cache.time.dt.month.values.astype(int)-1
    background = (cache.tropical_mean_sst.values.astype("float64")
                  - climate.tropical_sst_climatology.values[months].astype("float64"))
    normal = climate.tos_climatology.values[:,mask]
    absolute = cache.tos.values[:,mask].astype("float64")
    if not np.isfinite(absolute).all() or not np.isfinite(background).all():
        raise ValueError("Missing data on the frozen training ocean mask; repair the input without refitting on future years.")
    return absolute-normal[months]-background[:,None], background


def make_features(scores, background, origins, target_months, k, lookback):
    history = origins[:,None] - np.arange(lookback-1, -1, -1)
    seasonal = 2*np.pi*(np.asarray(target_months)-1)/12
    return np.column_stack((scores[history,:k].reshape(len(origins), lookback*k),
                            background[history], np.sin(seasonal), np.cos(seasonal)))


def decode_prediction(prediction, pca, k, sqrt_weights, initial_background):
    relative = (prediction[:,:k] @ pca.components_[:k] + pca.mean_) / sqrt_weights
    background = initial_background + prediction[:,k]
    return relative+background[:,None], background


def nino_weights(grid, mask):
    box = Region(190, 240, -5, 5)
    inside = box.mask(grid.lat, grid.lon).values
    weights = np.where(inside[mask], grid.areacello.values[mask], 0).astype("float64")
    if weights.sum() <= 0:
        raise ValueError("The training ocean mask has no Niño3.4 cells.")
    return weights/weights.sum()


def make_forecast_dataset(bundle, initialization_times, target_times, splits, anomaly_prediction,
                          background_prediction, truth=None, background_truth=None):
    """Store absolute SST; calculate indices from the same rounded SST fields."""
    grid, climate, mask = bundle["grid"], bundle["climatology"], bundle["mask"]
    spatial = grid.lat.dims
    time = xr.DataArray(target_times, dims="time", name="time")
    months = time.dt.month.values.astype(int)-1
    normal = climate.tos_climatology.values[months][:,mask]
    vectors = (anomaly_prediction+normal[None,:,:]).astype("float32")
    maps = np.full((2,len(target_times),*mask.shape), np.nan, dtype="float32")
    maps[:,:,mask] = vectors
    weight = nino_weights(grid, mask)
    # Subtract the saved climatology from the saved float32 temperatures first.
    indices = ((vectors-normal[None,:,:]).astype("float64") @ weight)
    coords = {"method": list(METHODS), "time": target_times,
              "initialization_time": ("time", initialization_times), "split": ("time", splits),
              "lat": grid.lat, "lon": grid.lon}
    coords.update({dim: grid[dim] for dim in spatial if dim in grid.coords})
    ds = xr.Dataset({
        "sst_prediction": (("method","time",*spatial), maps),
        "nino34_prediction": (("method","time"), indices),
        "relative_nino34_prediction": (("method","time"), indices-background_prediction),
        "tropical_ssta_prediction": (("method","time"), background_prediction),
        "areacello": grid.areacello, "model_mask": (spatial, mask.astype("int8")),
        "tos_climatology": climate.tos_climatology,
    }, coords=coords)
    if truth is not None:
        truth = np.asarray(truth, dtype="float32")
        truth_maps = np.full((len(target_times),*mask.shape), np.nan, dtype="float32")
        truth_maps[:,mask] = truth
        true_index = (truth-normal).astype("float64") @ weight
        ds["sst_truth"] = (("time",*spatial), truth_maps)
        ds["nino34_truth"] = ("time", true_index)
        ds["relative_nino34_truth"] = ("time", true_index-background_truth)
        ds["tropical_ssta_truth"] = ("time", background_truth)
    for name in ds.data_vars:
        if name not in {"areacello", "model_mask"}:
            ds[name].attrs["units"] = "degC"
    ds.sst_prediction.attrs["long_name"] = "Absolute SST forecast at the target month"
    ds.attrs = dict(grid.attrs)
    ds.attrs.update(lead_months=bundle["config"]["lead"], lookback_months=bundle["config"]["lookback"],
                    base_start=climate.attrs["base_start"], base_end=climate.attrs["base_end"],
                    train_end=bundle["config"]["train_end"], validation_end=bundle["config"]["validation_end"],
                    selected_components=bundle["selected_components"], selected_alpha=bundle["selected_alpha"],
                    split_rule="Targets within split; validation/test initialization strictly after previous cutoff",
                    index_definition="Monthly area-weighted Niño3.4 SSTA derived from saved SST on frozen model_mask",
                    relative_index_definition="Niño3.4 SSTA minus separately forecast tropical SSTA; no official RONI scaling",
                    persistence_definition="Full current SST anomaly held constant; add target-month training climatology",
                    selection_metric="Validation area-weighted SST RMSE only",
                    prediction_kind="CMIP simulation forecast; no observational forecast has been issued")
    return ds


def prepare_training_data(cache: xr.Dataset, config: ForecastConfig):
    """Fit the training-only spatial transforms once, independently of lead."""
    config.validate()
    cache = select_months(cache, config.train_start, config.test_end).load()
    validate_cache(cache)
    if "region_bounds" not in cache.attrs:
        raise ValueError("Cache is missing geographic region metadata; use data.py first.")
    region = Region(**json.loads(cache.attrs["region_bounds"]))
    if not bool(region.mask(xr.DataArray([-5.,5.]),xr.DataArray([190.,240.])).all()):
        raise ValueError("Cached region must contain the complete Niño3.4 box.")
    codes = validate_monthly_time(cache.time)
    train_months = codes <= month_number(config.train_end)
    base_end = config.base_end or config.train_end
    base_start = config.base_start or month_string(max(codes[0], month_number(base_end)-359))
    LOG.info("Fitting climatology on %s through %s (training cutoff %s)", base_start, base_end, config.train_end)
    climate = fit_climatology(cache, base_start, base_end, fit_until=config.train_end).compute()
    mask = (np.isfinite(cache.tos.values[train_months]).all(axis=0)
            & np.isfinite(climate.tos_climatology.values).all(axis=0)
            & np.isfinite(cache.areacello.values) & (cache.areacello.values > 0)
            & (cache.pacific_mask.values == 1))
    if not mask.any():
        raise ValueError("No consistently valid training ocean cells.")
    # The mask is determined exclusively from training data and is never refitted.
    relative, background = relative_vectors(cache, climate, mask)
    area = cache.areacello.values[mask].astype("float64")
    sqrt_weights = np.sqrt(area/area.mean())
    candidates = sorted(set(k for k in config.components if k <= min(train_months.sum()-1,mask.sum())))
    if not candidates:
        raise ValueError("PCA component candidates exceed the training matrix size.")
    if len(candidates) != len(set(config.components)):
        LOG.warning("Ignoring PCA candidates exceeding the training matrix size; using %s", candidates)
    weighted = relative*sqrt_weights
    LOG.info("Fitting area-weighted PCA: %s training months, %s ocean cells, up to %s components",
             train_months.sum(), mask.sum(), max(candidates))
    pca = PCA(n_components=max(candidates), svd_solver="randomized", random_state=config.seed)
    pca.fit(weighted[train_months])
    scores = pca.transform(weighted)
    del weighted
    return {"cache": cache, "climate": climate, "mask": mask,
            "relative": relative, "background": background, "area": area,
            "sqrt_weights": sqrt_weights, "pca": pca, "scores": scores,
            "candidates": candidates, "train_months": train_months,
            "preparation_config": asdict(config)}


def train_model(cache: xr.Dataset, config: ForecastConfig, *, prepared=None,
                validation_targets=None, test_targets=None):
    """Fit a lead-specific model, optionally reusing frozen training transforms.

    Target-month filters support a common verification window across leads.
    Excluded verification samples never participate in fitting or tuning.
    """
    config.validate()
    prepared = prepare_training_data(cache, config) if prepared is None else prepared
    for key in ("train_start", "train_end", "validation_end", "test_end",
                "base_start", "base_end", "components", "seed"):
        if prepared["preparation_config"][key] != asdict(config)[key]:
            raise ValueError(f"Prepared training data does not match {key}.")
    cache, climate, mask = (prepared[key] for key in ("cache", "climate", "mask"))
    relative, background, area = (prepared[key] for key in ("relative", "background", "area"))
    sqrt_weights, pca, scores = (prepared[key] for key in ("sqrt_weights", "pca", "scores"))
    candidates, train_months = prepared["candidates"], prepared["train_months"]
    base_start, base_end = climate.attrs["base_start"], climate.attrs["base_end"]
    origins, targets, split, samples = sample_splits(cache.time, config)
    codes = validate_monthly_time(cache.time)
    for name, allowed in (("validation", validation_targets), ("test", test_targets)):
        if allowed is not None:
            split[(split == name) & ~np.isin(codes[targets], allowed)] = "excluded"
        if np.count_nonzero(split == name) < 2:
            raise ValueError(f"Not enough common {name} target months.")
    samples["split"] = split
    train = split == "train"
    validation = split == "validation"
    valid_target_anomaly = relative[targets[validation]]+background[targets[validation],None]
    weights = area/area.sum()
    best, rows = None, []
    target_months = cache.time.dt.month.values[targets]
    for k in candidates:
        features = make_features(scores, background, origins, target_months, k, config.lookback)
        response = np.column_stack((scores[targets,:k], background[targets]-background[origins]))
        for alpha in sorted(set(config.alphas)):
            regressor = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="cholesky"))
            regressor.fit(features[train], response[train])
            prediction = regressor.predict(features[validation])
            predicted, predicted_background = decode_prediction(prediction, pca, k, sqrt_weights, background[origins[validation]])
            error = predicted-valid_target_anomaly
            rmse = float(np.sqrt(np.mean((error**2) @ weights)))
            row = {"components": k, "alpha": float(alpha), "validation_sst_rmse_degC": rmse,
                   "validation_tropical_rmse_degC": float(np.sqrt(np.mean((predicted_background-background[targets[validation]])**2))),
                   "pca_explained_variance_fraction": float(pca.explained_variance_ratio_[:k].sum())}
            rows.append(row)
            LOG.info("Validation: components=%s alpha=%g SST RMSE=%.4f °C", k, alpha, rmse)
            if best is None or rmse < best[0]:
                best = (rmse, k, float(alpha), regressor)
    _, k, alpha, regressor = best
    scores_table = pd.DataFrame(rows)
    scores_table["selected"] = (scores_table.components == k) & (scores_table.alpha == alpha)
    grid = cache[["areacello", "pacific_mask"]].drop_encoding()
    bundle = {"format_version": 1, "config": asdict(config), "climatology": climate.drop_encoding(),
              "grid": grid, "mask": mask, "sqrt_area_weights": sqrt_weights, "pca": pca,
              "regressor": regressor, "selected_components": k, "selected_alpha": alpha}
    LOG.info("Selected components=%s alpha=%g; generating validation and held-out forecasts ...", k, alpha)
    selected = (split == "validation") | (split == "test")
    selected_origins, selected_targets = origins[selected], targets[selected]
    features = make_features(scores, background, selected_origins,
                              cache.time.dt.month.values[selected_targets], k, config.lookback)
    predicted, predicted_background = decode_prediction(regressor.predict(features), pca, k, sqrt_weights,
                                                         background[selected_origins])
    baseline = relative[selected_origins]+background[selected_origins,None]
    forecasts = make_forecast_dataset(bundle, cache.time.values[selected_origins], cache.time.values[selected_targets],
                                      split[selected], np.stack([baseline,predicted]),
                                      np.stack([background[selected_origins],predicted_background]),
                                      truth=cache.tos.values[selected_targets][:,mask],
                                      background_truth=background[selected_targets])
    nino_region = Region(190,240,-5,5).mask(grid.lat,grid.lon).values
    original_nino_area = np.where(nino_region & (grid.pacific_mask.values == 1)
                                  & np.isfinite(grid.areacello.values) & (grid.areacello.values > 0), grid.areacello.values,0).sum(dtype="float64")
    kept_nino_area = grid.areacello.values[mask & nino_region].sum(dtype="float64")
    bundle["metadata"] = {
        "config": asdict(config), "base_start": base_start, "base_end": base_end,
        "selected_components": k, "selected_alpha": alpha, "validation_selection_rmse_degC": best[0],
        "sample_counts": {str(name): int((split == name).sum()) for name in np.unique(split)},
        "training_months_for_pca": int(train_months.sum()), "ocean_cells": int(mask.sum()),
        "nino34_retained_area_fraction": float(kept_nino_area/original_nino_area),
        "mask": "Finite in every training month and climatology month; positive area; frozen thereafter",
        "pca": "Training relative SSTA multiplied by sqrt(cell area / mean cell area); PCA centers training data",
        "features": "Past lookback months of retained PC scores and tropical SSTA, plus known target-month sin/cos",
        "targets": "Target relative PC scores and tropical SSTA increment from initialization to target",
        "standardization": "Feature StandardScaler fitted on supervised training samples only",
        "selection": "Minimum validation area-weighted SST RMSE; no refit on validation/test",
        "split_boundaries": "Validation/test origins strictly after previous cutoff; samples straddling cutoffs are purged",
        "versions": {name: version(name) for name in ("numpy","scikit-learn","joblib","xarray")},
        "scope": "Simulation evaluation only; actual-world forecast requires the instructor hindcast data and compatible preprocessing",
    }
    return bundle, forecasts, scores_table, samples


def predict_history(bundle: dict, history: xr.Dataset) -> xr.Dataset:
    """Forecast from the final lookback months of an already aligned cache.

    Uses saved training parameters only. The forecast month must follow the
    hyperparameter-selection cutoff; target time is labeled at month start.
    """
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported saved model format.")
    config = ForecastConfig(**bundle["config"])
    history = history.isel(time=slice(-config.lookback,None)).load()
    validate_cache(history)
    if history.sizes["time"] != config.lookback:
        raise ValueError(f"Need {config.lookback} months of history.")
    codes = validate_monthly_time(history.time)
    if codes[-1] <= month_number(config.validation_end):
        raise ValueError("Forecast initialization must follow the model selection cutoff.")
    for name in ("lat","lon"):
        if history[name].dims != bundle["grid"][name].dims or not np.allclose(history[name],bundle["grid"][name],atol=1e-5,rtol=0):
            raise ValueError(f"History {name} must be aligned with the saved model grid.")
    relative, background = relative_vectors(history,bundle["climatology"],bundle["mask"])
    target_code = int(codes[-1])+config.lead
    year, month = target_code//12, target_code%12+1
    reference = history.time.values[-1]
    if isinstance(reference,np.datetime64):
        target = np.datetime64(f"{year:04d}-{month:02d}-01")
    else:
        target = reference.replace(year=year,month=month,day=1,hour=0,minute=0,second=0,microsecond=0)
    origins = np.array([config.lookback-1])
    k = bundle["selected_components"]
    with threadpool_limits(limits=config.threads):
        scores = bundle["pca"].transform(relative*bundle["sqrt_area_weights"])
        features = make_features(scores,background,origins,np.array([month]),k,config.lookback)
        predicted, bg = decode_prediction(bundle["regressor"].predict(features),bundle["pca"],k,
                                           bundle["sqrt_area_weights"],background[origins])
    baseline = relative[origins]+background[origins,None]
    return make_forecast_dataset(bundle,np.array([reference]),np.array([target]),np.array(["forecast"]),
                                  np.stack([baseline,predicted]),np.stack([background[origins],bg]))


def write_indices(forecasts: xr.Dataset, path: Path):
    rows = []
    for method in forecasts.method.values:
        selected = forecasts.sel({"method":method})
        for i,(target,origin) in enumerate(zip(labels(forecasts.time),labels(forecasts.initialization_time))):
            row = {"method": str(method), "split": str(forecasts.split.values[i]),
                   "initialization_month": origin, "target_month": target}
            for name in ("nino34_prediction","relative_nino34_prediction","tropical_ssta_prediction",
                         "nino34_truth","relative_nino34_truth","tropical_ssta_truth"):
                if name in selected:
                    row[name] = float(selected[name].values[i])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path,index=False,float_format="%.10g")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--input",type=Path,default=ROOT / "data/raw" / f"{MODEL}_historical.nc")
    parser.add_argument("--output-dir",type=Path,default=ROOT / "data/model")
    for name,default in (("train-start","1850-01"),("train-end","1979-12"),("validation-end","1994-12"),("test-end","2014-12")):
        parser.add_argument("--"+name,default=default)
    parser.add_argument("--base-start")
    parser.add_argument("--base-end")
    parser.add_argument("--lookback",type=int,default=6)
    parser.add_argument("--lead",type=int,default=3)
    parser.add_argument("--components",default="10,20,40,80",help="Comma-separated PCA candidates")
    parser.add_argument("--alphas",default="0.1,1,10,100",help="Comma-separated positive ridge penalties")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    try:
        config = ForecastConfig(train_start=args.train_start,train_end=args.train_end,validation_end=args.validation_end,
                                test_end=args.test_end,base_start=args.base_start,base_end=args.base_end,
                                lookback=args.lookback,lead=args.lead,components=tuple(map(int,args.components.split(','))),
                                alphas=tuple(map(float,args.alphas.split(','))),seed=args.seed,threads=args.threads)
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    started = perf_counter()
    LOG.info("Loading SST cache %s",args.input)
    with xr.open_dataset(args.input,chunks=None) as cache, threadpool_limits(limits=config.threads):
        bundle, forecasts, scores, samples = train_model(cache,config)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    bundle["metadata"]["input_file"] = str(args.input.resolve())
    LOG.info("Saving model, forecast SST fields, indices, and audit tables ...")
    with tempfile.TemporaryDirectory(prefix=".model-",dir=args.output_dir) as staging:
        staging = Path(staging)
        joblib.dump(bundle,staging/"model.joblib",compress=3)
        write_netcdf(bundle["climatology"],staging/"training_climatology.nc")
        write_netcdf(forecasts,staging/"forecasts.nc")
        write_indices(forecasts,staging/"forecast_indices.csv")
        scores.to_csv(staging/"validation_scores.csv",index=False,float_format="%.10g")
        samples.to_csv(staging/"sample_splits.csv",index=False)
        (staging/"training_metadata.json").write_text(json.dumps(bundle["metadata"],indent=2,allow_nan=False)+"\n")
        for name in OUTPUTS:
            (staging/name).replace(args.output_dir/name)
    LOG.info("Saved %s; sample counts %s",args.output_dir,bundle["metadata"]["sample_counts"])
    LOG.info("Finished in %.2f s",perf_counter()-started)


if __name__ == "__main__":
    main()
