"""Compute SST anomalies and ENSO indices from the cache created by data.py.

The default 1985-2014 climatology reproduces the notebook's descriptive analysis.
For forecasting, fit a climatology on training years, then reuse it unchanged via
--climatology. Centered three-month indices are descriptive, NOT causal features.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import xarray as xr

from data import ROOT, MODEL, Region, area_mean, month_number, select_months, validate_monthly_time, write_netcdf

LOG = logging.getLogger(__name__)
NINO_BOXES = {
    "nino12": Region(270, 280, -10, 0),
    "nino3": Region(210, 270, -5, 5),
    "nino34": Region(190, 240, -5, 5),
    "nino4": Region(160, 210, -5, 5),
}
REQUIRED = {"tos", "areacello", "pacific_mask", "tropical_mean_sst"}


def validate_cache(ds: xr.Dataset):
    missing = REQUIRED - set(ds.data_vars)
    if missing:
        raise ValueError(f"Cache is missing {sorted(missing)}; create it using data.py.")
    validate_monthly_time(ds.time)
    if ds.tos.lat.ndim != 2 or ds.tos.lat.dims != ds.tos.lon.dims:
        raise ValueError("Expected native-grid two-dimensional lat/lon.")
    spatial = ds.tos.lat.dims
    if ds.tos.dims != ("time", *spatial):
        raise ValueError("SST dimensions must be time followed by the two grid dimensions.")
    if ds.areacello.dims != spatial or ds.pacific_mask.dims != spatial:
        raise ValueError("Area and region mask must be on the SST spatial grid.")
    if ds.tropical_mean_sst.dims != ("time",):
        raise ValueError("Global tropical SST must be a one-dimensional time series.")
    if ds.tos.attrs.get("units") != "degC" or ds.tropical_mean_sst.attrs.get("units") != "degC":
        raise ValueError("Cache temperatures must be in degC; use data.py for conversion.")
    if not bool(np.isfinite(ds.tropical_mean_sst).all().compute()):
        raise ValueError("Global tropical mean has missing months; inspect input coverage.")


def fit_climatology(ds: xr.Dataset, base_start: str = "1985-01",
                    base_end: str = "2014-12", fit_until: str | None = None) -> xr.Dataset:
    """Fit 12 monthly maps and the matching tropical climatology on explicit dates."""
    validate_cache(ds)
    if fit_until and month_number(base_end) > month_number(fit_until):
        raise ValueError("Climatology base_end exceeds --fit-until (training-data leakage).")
    base = select_months(ds, base_start, base_end)
    counts = np.bincount(base.time.dt.month.values, minlength=13)[1:]
    if (counts == 0).any():
        raise ValueError("Climatology requires observations of all 12 calendar months.")
    climate = xr.Dataset({
        "tos_climatology": base.tos.groupby("time.month").mean("time", skipna=True),
        "tropical_sst_climatology": base.tropical_mean_sst.groupby("time.month").mean("time"),
        "sample_count": ("month", counts),
    })
    for name in ("tos_climatology", "tropical_sst_climatology"):
        climate[name].attrs = {"units": "degC", "long_name": name.replace("_", " ")}
    climate.attrs = dict(ds.attrs)
    climate.attrs.update(base_start=base_start, base_end=base_end,
                         fit_until=fit_until or "not specified; descriptive analysis",
                         climatology_method="Arithmetic mean by calendar month")
    return climate


def compute_anomalies(ds: xr.Dataset, climate: xr.Dataset) -> xr.Dataset:
    """Apply an existing climatology without fitting or using future SST."""
    validate_cache(ds)
    # HDF5 may store lat/lon and SST with different spatial chunk sizes.
    # GroupBy arithmetic requires the field and its coordinates to agree.
    ds = ds.unify_chunks()
    for variable in ("tos_climatology", "tropical_sst_climatology"):
        if variable not in climate:
            raise ValueError(f"Climatology is missing {variable}.")
        if climate[variable].attrs.get("units") != "degC":
            raise ValueError("Climatology must be in degC.")
    if not np.array_equal(climate.month.values, np.arange(1, 13)):
        raise ValueError("Climatology must contain months 1 through 12 in order.")
    if climate.tos_climatology.dims != ("month", *ds.lat.dims):
        raise ValueError("Climatology map dimensions must be month followed by the SST grid.")
    if climate.tropical_sst_climatology.dims != ("month",):
        raise ValueError("Tropical climatology must have only the month dimension.")
    if not bool(np.isfinite(climate.tropical_sst_climatology).all().compute()):
        raise ValueError("Tropical climatology must be finite for every month.")
    for attr in ("source_id", "grid_label", "region_bounds", "tropical_bounds"):
        if attr in ds.attrs and attr in climate.attrs and ds.attrs[attr] != climate.attrs[attr]:
            raise ValueError(f"Climatology/input provenance differs for {attr}.")
    xr.align(ds.tos, climate.tos_climatology, join="exact")
    for name in ("lat", "lon"):
        if name not in climate.coords or not ds[name].equals(climate[name]):
            raise ValueError(f"Climatology/input {name} grids differ.")
    ssta = (ds.tos.groupby("time.month") - climate.tos_climatology).drop_vars("month")
    tropical = (ds.tropical_mean_sst.groupby("time.month")
                - climate.tropical_sst_climatology).drop_vars("month")
    ssta.attrs = {"units": "degC", "long_name": "SST anomaly relative to monthly climatology"}
    tropical.attrs = {"units": "degC", "long_name": "Global 20S-20N ocean mean SST anomaly"}
    relative = ssta - tropical
    relative.attrs = {"units": "degC", "long_name": "SST anomaly minus global tropical SST anomaly",
                      "note": "Relative anomaly; not a linear detrend or official RONI normalization"}
    result = xr.Dataset({"ssta": ssta, "ssta_relative": relative,
                         "tropical_ssta": tropical, "areacello": ds.areacello,
                         "pacific_mask": ds.pacific_mask})
    result.attrs = dict(ds.attrs)
    for key in ("base_start", "base_end", "fit_until", "climatology_method"):
        if key in climate.attrs:
            result.attrs[key] = climate.attrs[key]
    for key in ("source_id", "member_id", "experiment_id", "tos_zstore"):
        if key in climate.attrs:
            result.attrs[f"climatology_{key}"] = climate.attrs[key]
    return result


def _check_box_coverage(anomalies: xr.Dataset, box: Region, name: str):
    # The crop can contain cells from a box without containing the whole box.
    # Require recorded selection bounds, rather than silently averaging a partial region.
    if "region_bounds" not in anomalies.attrs:
        raise ValueError("Missing region_bounds metadata; cannot verify complete Niño regions.")
    region = Region(**json.loads(anomalies.attrs["region_bounds"]))
    points_lat = xr.DataArray([box.lat_min, box.lat_max])
    points_lon = xr.DataArray([box.lon_min, box.lon_max])
    if not bool(region.mask(points_lat, points_lon).all()):
        raise ValueError(f"Cached Pacific region does not include the complete {name} box.")


def compute_indices(anomalies: xr.Dataset) -> xr.Dataset:
    """Monthly Niño indices and explicitly labeled centered/trailing smooths.

    RONI-style here means Niño3.4 SSTA minus tropical SSTA. The official RONI
    variance rescaling and operational changing base periods are not implemented.
    """
    validate_monthly_time(anomalies.time)
    output = {}
    for name, box in NINO_BOXES.items():
        _check_box_coverage(anomalies, box, name)
        mask = box.mask(anomalies.lat, anomalies.lon) & (anomalies.pacific_mask == 1)
        index = area_mean(anomalies.ssta, anomalies.areacello, mask)
        index.attrs = {"units": "degC", "long_name": f"Monthly {name} SST anomaly",
                       "box": json.dumps(vars(box))}
        output[name] = index
    # Compute the small time series together. For lazy callers, this shares the
    # anomaly graph instead of recomputing the full field for each validity check.
    monthly = xr.Dataset({**output, "tropical_ssta": anomalies.tropical_ssta}).compute()
    output = {name: monthly[name] for name in NINO_BOXES}
    for name, index in output.items():
        if not bool(np.isfinite(index).all()):
            raise ValueError(f"{name} has a month with no valid SST; inspect input coverage.")
    relative = output["nino34"] - monthly.tropical_ssta
    relative.attrs = {"units": "degC", "long_name": "Monthly relative Niño3.4 anomaly"}
    output["relative_nino34"] = relative
    output["tropical_ssta"] = monthly.tropical_ssta
    for prefix, series in (("oni", output["nino34"]), ("roni_style", relative)):
        for suffix, centered in (("centered", True), ("trailing", False)):
            smooth = series.rolling(time=3, center=centered, min_periods=3).mean()
            smooth.attrs = {"units": "degC", "long_name": f"{prefix} three-month {suffix} mean",
                            "window": "t-1,t,t+1; needs next month" if centered else "t-2,t-1,t"}
            output[f"{prefix}_{suffix}"] = smooth
    ds = xr.Dataset(output, attrs=dict(anomalies.attrs))
    ds.attrs.update(
        relative_index_definition="Niño3.4 SSTA minus global 20S-20N ocean mean SSTA",
        roni_caveat="RONI-style only: no official variance rescaling or moving base periods",
        smoothing_caveat="Centered series use t+1; do not use as causal forecast inputs",
        event_caveat="Threshold crossings alone do not establish an official ENSO event",
    )
    return ds


def write_indices_csv(indices: xr.Dataset, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = indices.compute()
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".csv",
                                     newline="", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.writer(handle)
            names = list(indices.data_vars)
            writer.writerow(["month", *names])
            for i, (year, month) in enumerate(zip(indices.time.dt.year.values,
                                                 indices.time.dt.month.values)):
                values = [float(indices[name].values[i]) for name in names]
                writer.writerow([f"{int(year):04d}-{int(month):02d}",
                                 *[f"{v:.10g}" if np.isfinite(v) else "" for v in values]])
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def plot_diagnostics(cache: xr.Dataset, anomalies: xr.Dataset,
                     indices: xr.Dataset, output_dir: Path):
    """Simple Pacific maps without a Cartopy/coastline download dependency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    month = str(cache.time.values[-1])[:7]
    for filename, field, title, cmap in (
        ("sst_example.png", cache.tos.isel(time=-1), "SST", "viridis"),
        ("ssta_example.png", anomalies.ssta.isel(time=-1), "SST anomaly", "RdBu_r"),
    ):
        fig, ax = plt.subplots(figsize=(10, 4))
        limits = {"vmin": -3, "vmax": 3} if "anomaly" in title else {}
        image = ax.pcolormesh(cache.lon, cache.lat, field.compute(), shading="auto",
                              cmap=cmap, **limits)
        fig.colorbar(image, ax=ax, label="°C")
        ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title=f"{title}: {month}")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    years = indices.time.dt.year + (indices.time.dt.month - 0.5) / 12
    for name in ("nino34", "relative_nino34"):
        ax.plot(years, indices[name].compute(), label=name, lw=0.8)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set(xlabel="Model year", ylabel="SST anomaly (°C)", title="Monthly ENSO indices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "enso_timeseries.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "raw" / f"{MODEL}_historical.nc")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--base-start", help="First climatology month; default 1985-01")
    parser.add_argument("--base-end", help="Last climatology month; default 2014-12")
    parser.add_argument("--fit-until", help="Training cutoff YYYY-MM; reject later baseline data")
    parser.add_argument("--climatology", type=Path, help="Apply a previously fitted climatology")
    parser.add_argument("--plots", action="store_true", help="Also save three diagnostic PNGs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    started = perf_counter()
    if args.climatology and (args.base_start or args.base_end):
        parser.error("--climatology cannot be combined with --base-start/--base-end")
    files = ["pacific_ssta.nc", "enso_indices.nc", "enso_indices.csv"]
    if not args.climatology:
        files.append("monthly_climatology.nc")
    if args.plots:
        files += ["figures/sst_example.png", "figures/ssta_example.png", "figures/enso_timeseries.png"]
    # The already-cropped historical cache is about 91 MiB uncompressed.
    # Load it once: monthly GroupBy on tiny Dask chunks creates a large task
    # graph and repeated index reductions otherwise reread the same data.
    LOG.info("Opening local cache %s", args.input)
    with xr.open_dataset(args.input, chunks=None) as cache:
        LOG.info("Loading %.1f MiB of uncompressed regional data into memory ...",
                 cache.nbytes / 2**20)
        cache.load()
        LOG.info("Loaded %s months in %.2f s", cache.sizes["time"], perf_counter() - started)
        validate_cache(cache)
        if args.climatology:
            LOG.info("Loading fitted climatology %s", args.climatology)
            with xr.open_dataset(args.climatology) as opened:
                climate = opened.load()
            if args.fit_until:
                base_end = climate.attrs.get("base_end")
                if not base_end or month_number(base_end) > month_number(args.fit_until):
                    parser.error("Loaded climatology extends beyond --fit-until or lacks baseline metadata")
        else:
            LOG.info("Fitting monthly climatology for %s through %s ...",
                     args.base_start or "1985-01", args.base_end or "2014-12")
            climate = fit_climatology(cache, args.base_start or "1985-01",
                                      args.base_end or "2014-12", args.fit_until).compute()
        LOG.info("Computing ordinary and relative SST anomaly fields ...")
        anomalies = compute_anomalies(cache, climate)
        LOG.info("Computing the four Nino indices and three-month means ...")
        indices = compute_indices(anomalies).compute()
        LOG.info("Saving anomaly fields to %s ...", args.output_dir / "pacific_ssta.nc")
        write_netcdf(anomalies, args.output_dir / "pacific_ssta.nc")
        LOG.info("Saving index NetCDF and CSV files ...")
        write_netcdf(indices, args.output_dir / "enso_indices.nc")
        write_indices_csv(indices, args.output_dir / "enso_indices.csv")
        if not args.climatology:
            LOG.info("Saving monthly climatology ...")
            write_netcdf(climate, args.output_dir / "monthly_climatology.nc")
        if args.plots:
            LOG.info("Generating diagnostic figures ...")
            plot_diagnostics(cache, anomalies, indices, args.output_dir / "figures")
    LOG.info("Saved %s", ", ".join(files))
    LOG.info("Finished in %.2f s", perf_counter() - started)


if __name__ == "__main__":
    main()
