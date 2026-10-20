"""Compare two PCA--ridge settings and two index baselines at 1--24 months.

Run: python compare.py
Uses data.py's local Pacific cache. Every curve uses identical held-out target
months across all leads; hyperparameters are chosen on common validation
targets only. Three-month SST fields are saved for both settings and full-field
persistence. The four-index LIM is explicitly an index-only baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from threadpoolctl import threadpool_limits

from brinda_model import ALPHAS, COMPONENTS, fit_single_frame, prepare_single_frame
from data import MODEL, ROOT, Region, month_number, validate_monthly_time, write_netcdf
from evaluate import correlation
from model import ForecastConfig, labels, nino_weights, prepare_training_data, sample_splits, train_model
from preprocess import NINO_BOXES

LOG = logging.getLogger(__name__)
METHODS = ("persistence", "lim", "single_frame_ridge", "lagged_relative_ridge")
FIELD_METHODS = ("persistence", "single_frame_ridge", "lagged_relative_ridge")
LABELS = {"persistence": "Persistence", "lim": "LIM (4 indices)",
          "single_frame_ridge": "Single-frame ridge (Brinda)",
          "lagged_relative_ridge": "Lagged relative ridge (Haibo)"}
COLORS = {"persistence": "#777777", "lim": "#c54743",
          "single_frame_ridge": "#3177b5", "lagged_relative_ridge": "#008477"}


def fit_lim(prepared, ordinary):
    """Fit z(t+1)=G z(t) to four unsmoothed, area-weighted training indices."""
    cache, mask, area = prepared["cache"], prepared["mask"], prepared["area"]
    region = Region(**json.loads(cache.attrs["region_bounds"]))
    columns = []
    for name, box in NINO_BOXES.items():
        if not bool(region.mask(xr.DataArray([box.lat_min, box.lat_max]),
                                xr.DataArray([box.lon_min, box.lon_max])).all()):
            raise ValueError(f"The comparison cache must contain the complete {name} box.")
        weights = np.where(box.mask(cache.lat, cache.lon).values[mask], area, 0)
        if weights.sum() <= 0:
            raise ValueError(f"No training ocean area for {name}.")
        columns.append(ordinary @ (weights / weights.sum()))
    indices = np.column_stack(columns)
    training = indices[prepared["train_months"]]
    # lstsq is numerically safer than explicitly inverting X0 X0^T.
    propagator = np.linalg.lstsq(training[:-1], training[1:], rcond=None)[0].T
    return indices, propagator


def forecast_dataset(prepared, haibo, single_anomaly, lim_prediction, lead):
    """Combine three field methods and four independently verifiable indices."""
    mask, climate = prepared["mask"], prepared["climate"]
    normal = climate.tos_climatology.values[haibo.time.dt.month.values - 1][:, mask]
    single = (single_anomaly + normal).astype("float32")
    shape = (len(haibo.time), *mask.shape)
    maps = np.full(shape, np.nan, dtype="float32")
    maps[:, mask] = single
    fields = np.stack([haibo.sst_prediction.sel({"method": "persistence"}).values,
                       maps, haibo.sst_prediction.sel({"method": "pca_ridge"}).values])
    weights = nino_weights(haibo, mask)
    derived = (fields[..., mask] - normal[None]).astype("float64") @ weights
    indices = np.stack([derived[0], lim_prediction, derived[1], derived[2]])
    spatial = haibo.lat.dims
    coords = {"method": list(METHODS), "field_method": list(FIELD_METHODS),
              "time": haibo.time, "initialization_time": haibo.initialization_time,
              "split": haibo.split, "lat": haibo.lat, "lon": haibo.lon}
    coords.update({name: haibo[name] for name in spatial if name in haibo.coords})
    ds = xr.Dataset({
        "sst_prediction": (("field_method", "time", *spatial), fields),
        "nino34_prediction": (("method", "time"), indices),
        "sst_truth": haibo.sst_truth, "nino34_truth": haibo.nino34_truth,
        "areacello": haibo.areacello, "model_mask": haibo.model_mask,
        "tos_climatology": climate.tos_climatology,
    }, coords=coords)
    ds.attrs = {name: value for name, value in haibo.attrs.items()
                if name not in {"selected_components", "selected_alpha", "selection_metric", "relative_index_definition", "lookback_months"}}
    ds.attrs.update(lead_months=lead,
                    single_frame_lookback_months=1, lagged_relative_lookback_months=haibo.attrs["lookback_months"],
                    lim_scope="Index-only baseline; no SST field is claimed for LIM",
                    comparison_scope="Shared Pacific domain, training climatology/mask, split cutoffs, and verification targets",
                    index_definition="Monthly area-weighted Nino3.4 SSTA; field methods derived from saved SST; LIM predicted directly",
                    single_frame_provenance="Adapted from Brinda Puri's notebook, cells 29-52; regional rather than global",
                    selection="Single-frame: validation Nino3.4 correlation; lagged-relative: validation area-weighted SST RMSE")
    for name in ("sst_prediction", "nino34_prediction", "sst_truth", "nino34_truth", "tos_climatology"):
        ds[name].attrs["units"] = "degC"
    validate_comparison(ds, prepared)
    return ds


def validate_comparison(ds, prepared):
    mask = ds.model_mask.values == 1
    if not np.isfinite(ds.sst_prediction.values[..., mask]).all():
        raise ValueError("Nonfinite field prediction on the frozen training mask.")
    if not np.isfinite(ds.nino34_prediction.values).all():
        raise ValueError("Nonfinite index prediction.")
    codes = ds.time.dt.year.values * 12 + ds.time.dt.month.values - 1
    origins = ds.initialization_time.dt.year.values * 12 + ds.initialization_time.dt.month.values - 1
    if not np.all(codes - origins == ds.attrs["lead_months"]):
        raise ValueError("Incorrect forecast lead.")
    for name, boundary in (("validation", "train_end"), ("test", "validation_end")):
        validate_monthly_time(ds.time.isel(time=np.flatnonzero(ds.split.values == name)))
        if np.any(origins[ds.split.values == name] <= month_number(ds.attrs[boundary])):
            raise ValueError("Forecast initialization overlaps a previous fitting/selection period.")
    normal = ds.tos_climatology.values[ds.time.dt.month.values - 1][:, mask]
    weights = nino_weights(ds, mask)
    for name in FIELD_METHODS:
        expected = (ds.sst_prediction.sel(field_method=name).values[:, mask] - normal).astype(float) @ weights
        np.testing.assert_allclose(ds.nino34_prediction.sel({"method": name}), expected, atol=2e-6, rtol=0)
    # Compare reference fields directly with the raw cache, not another index helper.
    source_codes = validate_monthly_time(prepared["cache"].time)
    locations = np.searchsorted(source_codes, codes)
    reference = prepared["cache"].tos.values[locations][:, mask].astype("float32")
    np.testing.assert_array_equal(ds.sst_truth.values[:, mask], reference)
    np.testing.assert_allclose(ds.nino34_truth, (reference - normal).astype(float) @ weights, atol=2e-6, rtol=0)


def summarize(ds):
    rows, records = [], []
    mask = ds.model_mask.values == 1
    weights = ds.areacello.values[mask].astype(float)
    weights /= weights.sum()
    dates, origins = labels(ds.time), labels(ds.initialization_time)
    for split in ("validation", "test"):
        selected = ds.split.values == split
        truth_index = ds.nino34_truth.values[selected]
        truth_fields = ds.sst_truth.values[selected][:, mask].astype(float)
        for method in METHODS:
            index = ds.nino34_prediction.sel({"method": method}).values[selected]
            row = {"lead_months": int(ds.attrs["lead_months"]), "split": split, "method": method,
                   "n_samples": int(selected.sum()), "target_start": np.array(dates)[selected][0],
                   "target_end": np.array(dates)[selected][-1],
                   "nino34_rmse_degC": float(np.sqrt(np.mean((index - truth_index)**2))),
                   "nino34_correlation": correlation(index, truth_index),
                   "nino34_bias_degC": float(np.mean(index - truth_index)), "sst_rmse_degC": np.nan}
            if method in FIELD_METHODS:
                field = ds.sst_prediction.sel(field_method=method).values[selected][:, mask].astype(float)
                row["sst_rmse_degC"] = float(np.sqrt(np.mean((field - truth_fields)**2 @ weights)))
            rows.append(row)
            for location, p, t in zip(np.flatnonzero(selected), index, truth_index):
                records.append({"lead_months": row["lead_months"], "split": split, "method": method,
                                "initialization_month": origins[location], "target_month": dates[location],
                                "nino34_prediction": float(p), "nino34_truth": float(t)})
    result = pd.DataFrame(rows)
    for split in ("validation", "test"):
        selected = result.split == split
        baseline = result.loc[selected & (result.method == "persistence")].iloc[0]
        for field in ("sst", "nino34"):
            result.loc[selected, field + "_mse_skill_vs_persistence"] = (
                1 - (result.loc[selected, field + "_rmse_degC"] / baseline[field + "_rmse_degC"])**2)
    return result, pd.DataFrame(records)


def make_figures(metrics, forecasts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in METHODS:
        rows = metrics[(metrics.split == "test") & (metrics.method == method)]
        ax.plot(rows.lead_months, rows.nino34_correlation, "o-", color=COLORS[method],
                label=LABELS[method], markersize=3)
    ax.axhline(0.5, color="0.55", ls=":")
    ax.axhline(0, color="0.8", lw=0.7)
    ax.set(xlabel="Forecast lead (months)", ylabel="Niño3.4 correlation (held out)",
           title="Two PCA–ridge settings and two baselines", xlim=(1, metrics.lead_months.max()), ylim=(-1, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "lead_skill.png", dpi=180)
    plt.close(fig)

    ds = forecasts.isel(time=np.flatnonzero(forecasts.split.values == "test")[-1])
    normal = ds.tos_climatology.sel(month=int(ds.time.dt.month))
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True, sharey=True)
    panels = [("Reference", ds.sst_truth)] + [(LABELS[m], ds.sst_prediction.sel(field_method=m)) for m in FIELD_METHODS]
    for ax, (name, values) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(ds.lon, ds.lat, values - normal, shading="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set(title=name, xlim=(120, 290), ylim=(-20, 20))
    fig.colorbar(mesh, ax=axes, label="SST anomaly (°C)", shrink=0.8, pad=0.02)
    fig.supxlabel("Longitude (°E)")
    fig.supylabel("Latitude (°N)")
    date = f"{int(ds.time.dt.year):04d}-{int(ds.time.dt.month):02d}"
    fig.suptitle(f"{forecasts.attrs['lead_months']}-month forecast: final test target {date}")
    fig.savefig(output / "sst_forecast_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_comparison(cache, config, max_lead, field_lead, output):
    if not 1 <= field_lead <= max_lead:
        raise ValueError("Require 1 <= field_lead <= max_lead.")
    prepared = prepare_training_data(cache, config)
    cache = prepared["cache"]
    codes = validate_monthly_time(cache.time)
    _, targets_max, split_max, _ = sample_splits(cache.time, replace(config, lead=max_lead))
    allowed = {name: codes[targets_max[split_max == name]] for name in ("validation", "test")}
    LOG.info("Common validation targets: %s–%s (%d); test: %s–%s (%d)",
             labels(cache.time.isel(time=np.searchsorted(codes, allowed["validation"][[0, -1]])))[0],
             labels(cache.time.isel(time=np.searchsorted(codes, allowed["validation"][[0, -1]])))[1], len(allowed["validation"]),
             labels(cache.time.isel(time=np.searchsorted(codes, allowed["test"][[0, -1]])))[0],
             labels(cache.time.isel(time=np.searchsorted(codes, allowed["test"][[0, -1]])))[1], len(allowed["test"]))
    single_context = prepare_single_frame(prepared, config.seed)
    indices, propagator = fit_lim(prepared, single_context["ordinary"])
    weight = nino_weights(cache, prepared["mask"])
    nino_column = list(NINO_BOXES).index("nino34")
    tables, index_tables, score_tables, split_tables, settings = [], [], [], [], []
    field_forecasts, saved = None, None
    for lead in range(1, max_lead + 1):
        LOG.info("Lead %d/%d: fitting the two settings on common verification targets ...", lead, max_lead)
        lead_config = replace(config, lead=lead)
        bundle, haibo, scores, samples = train_model(cache, lead_config, prepared=prepared,
                                                     validation_targets=allowed["validation"], test_targets=allowed["test"])
        origins, targets, split, _ = sample_splits(cache.time, lead_config)
        split[:] = samples.split.to_numpy()
        single_bundle, single_prediction, single_scores = fit_single_frame(single_context, origins, targets, split, weight)
        selected = (split == "validation") | (split == "test")
        lim_prediction = (indices[origins[selected]] @ np.linalg.matrix_power(propagator, lead).T)[:, nino_column]
        ds = forecast_dataset(prepared, haibo, single_prediction, lim_prediction, lead)
        metrics, predictions = summarize(ds)
        tables.append(metrics)
        index_tables.append(predictions)
        for name, table in (("lagged_relative_ridge", scores), ("single_frame_ridge", single_scores)):
            score_tables.append(table.assign(method=name, lead_months=lead))
        split_tables.append(samples.assign(lead_months=lead))
        settings.append({"lead_months": lead,
                         "single_frame_components": single_bundle["selected_components"],
                         "single_frame_alpha": single_bundle["selected_alpha"],
                         "lagged_relative_components": bundle["selected_components"],
                         "lagged_relative_alpha": bundle["selected_alpha"]})
        if lead == field_lead:
            field_forecasts = ds
            saved = {"format_version": 1, "lagged_relative": bundle, "single_frame": single_bundle,
                     "lim_propagator": propagator, "lim_index_names": list(NINO_BOXES),
                     "shared_preprocessing": {key: prepared[key] for key in ("climate", "mask")}}
        brief = metrics[metrics.split == "test"]
        LOG.info("Lead %d test r: %s", lead, "; ".join(f"{r.method}={r.nino34_correlation:.3f}" for r in brief.itertuples()))

    metrics = pd.concat(tables, ignore_index=True)
    for name in ("validation", "test"):
        part = metrics[metrics.split == name]
        if part.n_samples.nunique() != 1 or part.target_start.nunique() != 1 or part.target_end.nunique() != 1:
            raise AssertionError("Methods/leads have different verification windows.")
    metadata = {"config": asdict(config), "max_lead_months": max_lead, "saved_field_lead_months": field_lead,
                "methods": LABELS, "field_methods": list(FIELD_METHODS),
                "common_targets": {name: {"start": metrics.loc[metrics.split == name, "target_start"].iloc[0],
                                          "end": metrics.loc[metrics.split == name, "target_end"].iloc[0],
                                          "count": int(metrics.loc[metrics.split == name, "n_samples"].iloc[0])} for name in allowed},
                "base_start": prepared["climate"].attrs["base_start"], "base_end": prepared["climate"].attrs["base_end"],
                "single_frame_source": "Brinda Puri, brinda/ECS289G_ENSO_project.ipynb, cells 29–52",
                "adaptations": ["Common Pacific domain instead of the notebook's global domain",
                                "Training-only climatology, frozen training ocean mask, and training-period PCA",
                                "Shared chronological cutoffs and common target months across all leads",
                                "Shared maximum-rank PCA with leading-component prefixes for each setting"],
                "single_frame_candidates": {"components": COMPONENTS, "alphas": ALPHAS},
                "selection": {"single_frame_ridge": "Maximum validation monthly Nino3.4 correlation",
                              "lagged_relative_ridge": "Minimum validation area-weighted SST RMSE"},
                "lim": {"definition": "Uncentered least-squares one-month propagator on four monthly training indices; iterate G^lead",
                        "spectral_radius": float(np.max(abs(np.linalg.eigvals(propagator)))), "scope": "Index-only baseline"},
                "target_consistency": "All methods and leads use identical target months, monthly unsmoothed indices, climatology and fixed mask",
                "history_note": "Only initialization dates must follow the preceding cutoff; known earlier input history remains causal",
                "scope": "Simulation comparison only; instructor-data prediction remains pending"}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=output) as staging:
        staging = Path(staging)
        write_netcdf(field_forecasts, staging / "forecasts.nc")
        joblib.dump(saved, staging / "models.joblib", compress=3)
        metrics.to_csv(staging / "metrics.csv", index=False, float_format="%.10g")
        pd.concat(index_tables, ignore_index=True).to_csv(staging / "forecast_indices.csv", index=False, float_format="%.10g")
        pd.concat(score_tables, ignore_index=True).to_csv(staging / "validation_scores.csv", index=False, float_format="%.10g")
        pd.concat(split_tables, ignore_index=True).to_csv(staging / "sample_splits.csv", index=False)
        pd.DataFrame(settings).to_csv(staging / "selected_settings.csv", index=False)
        (staging / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        make_figures(metrics, field_forecasts, staging / "figures")
        for path in staging.rglob('*'):
            if path.is_file():
                destination = output / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
    return metrics, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw" / f"{MODEL}_historical.nc")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/comparison")
    parser.add_argument("--max-lead", type=int, default=24)
    parser.add_argument("--field-lead", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_lead < 1 or not 1 <= args.field_lead <= args.max_lead or args.threads < 1:
        parser.error("Require max-lead >= field-lead >= 1 and threads >= 1.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Keep the original fitting implementation's per-candidate output concise here.
    logging.getLogger("model").setLevel(logging.WARNING)
    started = perf_counter()
    with xr.open_dataset(args.input, chunks=None) as cache, threadpool_limits(limits=args.threads):
        run_comparison(cache, ForecastConfig(threads=args.threads), args.max_lead, args.field_lead, args.output_dir)
    LOG.info("Saved comparison to %s; finished in %.2f s", args.output_dir, perf_counter() - started)


if __name__ == "__main__":
    main()
