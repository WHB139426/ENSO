"""Characterize simulated ENSO from the outputs of preprocess.py.

Run: python analysis.py
This is descriptive analysis, not a forecast or an observational model evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from scipy.integrate import trapezoid

from data import ROOT, select_months, validate_monthly_time, write_netcdf

LOG = logging.getLogger(__name__)
MONTHLY_INDICES = ("nino12", "nino3", "nino34", "nino4", "relative_nino34")
EVENT_COLUMNS = ["phase", "start_month", "end_month", "duration_months", "peak_month",
                 "peak_index_degC", "mean_index_degC", "left_censored", "right_censored"]
OUTPUTS = ("index_statistics.csv", "events.csv", "event_summary.csv", "spectrum.csv",
           "autocorrelation.csv", "cycle_summary.csv", "enso_composites.nc", "analysis_metadata.json",
           "figures/index_timeseries.png", "figures/enso_variability.png",
           "figures/event_statistics.png", "figures/enso_composites.png")


def month_labels(time: xr.DataArray) -> list[str]:
    return [f"{int(y):04d}-{int(m):02d}" for y, m in zip(time.dt.year.values, time.dt.month.values)]


def index_statistics(indices: xr.Dataset) -> pd.DataFrame:
    """Describe amplitude and linear trend without silently removing the trend."""
    rows = []
    dates = month_labels(indices.time)
    for name in (*MONTHLY_INDICES, "oni_centered", "roni_style_centered"):
        values = indices[name].values.astype("float64")
        positions = np.flatnonzero(np.isfinite(values))
        if positions.size < 2:
            raise ValueError(f"{name} requires at least two finite observations.")
        valid = values[positions]
        slope, intercept = np.polyfit(positions / 12, valid, 1)
        residual = valid - (intercept + slope * positions / 12)
        rows.append({"index": name, "n_months": len(valid), "mean_degC": valid.mean(),
                     "std_degC": valid.std(ddof=1), "detrended_std_degC": residual.std(ddof=1),
                     "min_degC": valid.min(), "min_month": dates[positions[valid.argmin()]],
                     "max_degC": valid.max(), "max_month": dates[positions[valid.argmax()]],
                     "linear_trend_degC_per_decade": slope * 10})
    return pd.DataFrame(rows)


def detect_events(series: xr.DataArray, threshold: float = 0.5,
                  min_duration: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """Find consecutive threshold runs; mark runs touching missing/record boundaries.

    Duration is the number of monthly centers of overlapping three-month means.
    A censored duration is an observed lower bound, not a complete event duration.
    """
    validate_monthly_time(series.time)
    if not np.isfinite(threshold) or threshold <= 0 or min_duration < 1:
        raise ValueError("Event threshold must be positive and min_duration >= 1.")
    values = np.asarray(series.values, dtype="float64")
    if values.shape != (series.sizes["time"],):
        raise ValueError("Event index must have only the time dimension.")
    dates = month_labels(series.time)
    state = np.zeros(len(values), dtype="int8")
    events = []
    for label, sign in (("warm", 1), ("cold", -1)):
        exceeds = np.isfinite(values) & (sign * values >= threshold)
        changes = np.diff(np.r_[False, exceeds, False].astype("int8"))
        for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if stop - start < min_duration:
                continue
            event_values = values[start:stop]
            peak = start + int(np.argmax(sign * event_values))
            events.append({"phase": label, "start_month": dates[start], "end_month": dates[stop-1],
                           "duration_months": int(stop-start), "peak_month": dates[peak],
                           "peak_index_degC": float(values[peak]), "mean_index_degC": float(event_values.mean()),
                           "left_censored": bool(start == 0 or not np.isfinite(values[start-1])),
                           "right_censored": bool(stop == len(values) or not np.isfinite(values[stop]))})
            state[start:stop] = sign
    frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if len(frame):
        frame = frame.sort_values("start_month").reset_index(drop=True)
    return frame, state


def event_statistics(events: pd.DataFrame, n_valid_months: int) -> pd.DataFrame:
    rows = []
    for phase in ("warm", "cold"):
        subset = events.loc[events.phase == phase]
        complete = subset.loc[~(subset.left_censored.astype(bool) | subset.right_censored.astype(bool))]
        intervals = np.diff([int(d[:4])*12 + int(d[5:]) for d in complete.peak_month]) / 12
        strongest = subset.loc[subset.peak_index_degC.abs().idxmax()] if len(subset) else None
        rows.append({"phase": phase, "n_events": len(subset), "n_complete_events": len(complete),
                     "n_censored_events": len(subset)-len(complete),
                     "event_months": int(subset.duration_months.sum()),
                     "fraction_of_valid_months": subset.duration_months.sum()/n_valid_months,
                     "events_per_decade": len(subset)/(n_valid_months/120),
                     "mean_observed_duration_months": subset.duration_months.mean(),
                     "max_observed_duration_months": subset.duration_months.max(),
                     "mean_complete_duration_months": complete.duration_months.mean(),
                     "mean_complete_peak_interval_years": float(intervals.mean()) if len(intervals) else None,
                     "strongest_peak_degC": strongest.peak_index_degC if strongest is not None else None,
                     "strongest_peak_month": strongest.peak_month if strongest is not None else None})
    return pd.DataFrame(rows)


def variability_analysis(indices: xr.Dataset, segment_years: int = 40,
                         max_lag: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Welch PSD and biased normalized ACF of linearly detrended monthly indices."""
    if segment_years < 2 or max_lag < 0:
        raise ValueError("Welch segment must be >= 2 years and max lag >= 0.")
    spectra, correlations, summaries = [], [], []
    for name in ("nino34", "relative_nino34"):
        values = indices[name].values.astype("float64")
        if len(values) < 24 or not np.isfinite(values).all():
            raise ValueError("Spectral analysis needs at least 24 continuous, finite monthly values.")
        detrended = signal.detrend(values, type="linear")
        nperseg = min(segment_years*12, len(values))
        frequency, power = signal.welch(detrended, fs=12, window="hann", nperseg=nperseg,
                                         noverlap=nperseg//2, detrend=False, scaling="density")
        periods = np.divide(1, frequency, out=np.full_like(frequency, np.nan), where=frequency > 0)
        spectra.append(pd.DataFrame({"index": name, "frequency_cycles_per_year": frequency,
                                     "period_years": periods, "psd_degC2_per_cycles_per_year": power}))
        lags = np.arange(min(max_lag, len(values)-1)+1)
        norm = float(np.dot(detrended, detrended))
        variable = values.std() > 0 and norm > np.finfo(float).eps * max(1., float(np.dot(values, values)))
        acf = [float(np.dot(detrended[:len(values)-lag], detrended[lag:])/norm)
               if variable else np.nan for lag in lags]
        correlations.append(pd.DataFrame({"index": name, "lag_months": lags, "autocorrelation": acf}))

        def peak(low, high):
            chosen = np.flatnonzero((periods >= low) & (periods <= high))
            if not variable or len(chosen) == 0:
                return None
            return float(periods[chosen[np.argmax(power[chosen])]])

        # Integrate the interpolated density at the exact 1/7 and 1/2 boundaries.
        band_frequency = np.r_[1/7, frequency[(frequency > 1/7) & (frequency < 1/2)], 1/2]
        band_density = np.interp(band_frequency, frequency, power)
        band_power = float(trapezoid(band_density, band_frequency))
        total_power = float(trapezoid(power, frequency))
        summaries.append({"index": name, "n_months": len(values), "segment_months": nperseg,
                          "frequency_resolution_cycles_per_year": 12/nperseg,
                          "peak_period_1_20_years": peak(1, 20), "peak_period_2_7_years": peak(2, 7),
                          "power_fraction_2_7_years": band_power/total_power if variable and total_power > 0 else None})
    return pd.concat(spectra, ignore_index=True), pd.concat(correlations, ignore_index=True), pd.DataFrame(summaries)


def event_composites(fields: xr.Dataset, events: pd.DataFrame) -> xr.Dataset:
    """Average months within each event, then give every event equal weight."""
    output = xr.Dataset()
    for name in ("ssta", "ssta_relative"):
        composites, counts = [], []
        for phase in ("warm", "cold"):
            maps = [fields[name].sel(time=slice(row.start_month, row.end_month)).mean("time", skipna=True)
                    for row in events.loc[events.phase == phase].itertuples()]
            if maps:
                stack = xr.concat(maps, dim="event")
                composites.append(stack.mean("event", skipna=True))
                counts.append(stack.notnull().sum("event"))
            else:
                template = fields[name].isel(time=0, drop=True)
                composites.append(xr.full_like(template, np.nan))
                counts.append(xr.zeros_like(template, dtype="int32"))
        phase_coord = pd.Index(["warm", "cold"], name="phase")
        output[f"{name}_composite"] = xr.concat(composites, dim=phase_coord)
        output[f"{name}_composite"].attrs = {"units": "degC", "weighting": "Equal weight per event"}
        output[f"{name}_valid_event_count"] = xr.concat(counts, dim=phase_coord)
    output["event_count"] = ("phase", [int((events.phase == phase).sum()) for phase in ("warm", "cold")])
    output["event_month_count"] = ("phase", [int(events.loc[events.phase == phase, "duration_months"].sum())
                                            for phase in ("warm", "cold")])
    output.attrs = dict(fields.attrs)
    output.attrs.update(composite_weighting="Mean over months within each event, then mean over events",
                         censored_events="Included; available event months only")
    return output


def write_table(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, float_format="%.10g")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_figures(indices, events, state, spectra, correlations, composites, output_dir,
                 event_index, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)
    years = indices.time.dt.year.values + (indices.time.dt.month.values-0.5)/12
    model = indices.attrs.get("source_id", "CMIP6")
    colors = {"nino34": "tab:blue", "relative_nino34": "tab:orange"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained")
    for name, color in colors.items():
        axes[0].plot(years, indices[name], lw=0.7, label=name, color=color)
    axes[0].set(title=f"{model}: monthly Niño3.4 anomalies", ylabel="Anomaly (°C)")
    axes[0].legend()
    selected = indices[event_index].values
    axes[1].plot(years, selected, color="0.25", lw=0.9, label=event_index)
    for sign, label, color in ((1, "Warm event", "tab:red"), (-1, "Cold event", "tab:blue")):
        axes[1].fill_between(years, 0, selected, where=state == sign, color=color, alpha=0.5, label=label)
    for value in (-threshold, threshold):
        axes[1].axhline(value, color="gray", ls=":", lw=0.8)
    axes[1].set(xlim=(max(years[0], years[-1]-30), years[-1]), xlabel="Model year",
                ylabel="Three-month anomaly (°C)", title="Qualified events: last 30 years (or available record)")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "index_timeseries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, color in colors.items():
        psd = spectra.loc[(spectra["index"] == name) & spectra.period_years.between(1, 20)].sort_values("period_years")
        axes[0].plot(psd.period_years, psd.psd_degC2_per_cycles_per_year, label=name, color=color)
        acf = correlations.loc[correlations["index"] == name]
        axes[1].plot(acf.lag_months, acf.autocorrelation, label=name, color=color)
    axes[0].axvspan(2, 7, color="gray", alpha=0.15, label="2–7 years")
    axes[0].set(xlabel="Period (years)", ylabel="PSD [°C² / (cycles/year)]", title="Welch spectrum: linearly detrended indices")
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set(xlabel="Lag (months)", ylabel="Normalized autocorrelation", title="Autocorrelation: linearly detrended indices")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "enso_variability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    counts = [int((events.phase == phase).sum()) for phase in ("warm", "cold")]
    axes[0].bar(["Warm", "Cold"], counts, color=["tab:red", "tab:blue"])
    axes[0].set(ylabel="Number of events", title="Qualified events (including censored)")
    maximum = int(events.duration_months.max()) if len(events) else 5
    bins = np.arange(0.5, maximum+1.5, 1)
    for phase, color in (("warm", "tab:red"), ("cold", "tab:blue")):
        duration = events.loc[events.phase == phase, "duration_months"].to_numpy(dtype=int)
        axes[1].hist(duration, bins=bins, alpha=0.5, label=phase, color=color)
    axes[1].set(xlabel="Observed duration (monthly centers)", ylabel="Number of events",
                title="Duration; boundary events can be incomplete")
    axes[1].legend()
    fig.suptitle(f"{model}: {event_index}, threshold ±{threshold}°C")
    fig.savefig(output_dir / "event_statistics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    finite = np.concatenate([composites[name].values.ravel() for name in ("ssta_composite", "ssta_relative_composite")])
    finite = finite[np.isfinite(finite)]
    limit = max(1., np.ceil(np.max(abs(finite))*2)/2) if len(finite) else 1.
    for row, phase in enumerate(("warm", "cold")):
        n = int(composites.event_count.sel(phase=phase))
        for col, (name, label) in enumerate((("ssta_composite", "SST anomaly"), ("ssta_relative_composite", "Relative SST anomaly"))):
            ax = axes[row, col]
            image = ax.pcolormesh(composites.lon, composites.lat, composites[name].sel(phase=phase),
                                  shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.add_patch(Rectangle((190, -5), 50, 10, fill=False, edgecolor="black", lw=1))
            ax.set(title=f"{phase.capitalize()}: {label} ({n} events)", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
            if n == 0:
                ax.text(0.5, 0.5, "No qualifying events", transform=ax.transAxes, ha="center")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Anomaly (°C)", shrink=0.85)
    fig.suptitle(f"{model}: events classified by {event_index}\nEqual weight per event; box = Niño3.4")
    fig.savefig(output_dir / "enso_composites.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=Path, default=ROOT / "data/processed/enso_indices.nc")
    parser.add_argument("--fields", type=Path, default=ROOT / "data/processed/pacific_ssta.nc")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/analysis")
    parser.add_argument("--start", help="First analysis month, YYYY-MM")
    parser.add_argument("--end", help="Last analysis month, YYYY-MM")
    parser.add_argument("--event-index", choices=("roni_style_centered", "oni_centered"), default="roni_style_centered")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration", type=int, default=5)
    parser.add_argument("--segment-years", type=int, default=40, help="Welch segment length, capped to record length")
    parser.add_argument("--max-lag", type=int, default=120, help="Maximum autocorrelation lag in months")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not np.isfinite(args.threshold) or args.threshold <= 0 or args.min_duration < 1:
        parser.error("--threshold must be positive and --min-duration >= 1")
    if args.segment_years < 2 or args.max_lag < 0:
        parser.error("--segment-years must be >= 2 and --max-lag >= 0")
    if not args.overwrite:
        existing = [str(args.output_dir / name) for name in OUTPUTS if (args.output_dir / name).exists()]
        if existing:
            parser.error(f"Outputs exist: {existing}; use --overwrite or another --output-dir")
    started = perf_counter()
    LOG.info("Loading preprocessed indices and anomaly fields ...")
    with xr.open_dataset(args.indices, chunks=None) as source:
        indices = select_months(source, args.start, args.end).load()
    with xr.open_dataset(args.fields, chunks=None) as source:
        fields = select_months(source, args.start, args.end).load()
    if not indices.time.equals(fields.time):
        raise ValueError("Indices and anomaly fields must cover exactly the same months.")
    for attr in ("source_id", "member_id", "experiment_id", "base_start", "base_end"):
        if indices.attrs.get(attr) != fields.attrs.get(attr):
            raise ValueError(f"Index/field provenance differs for {attr}.")
    for name in (*MONTHLY_INDICES, "oni_centered", "roni_style_centered"):
        if name not in indices or indices[name].dims != ("time",) or indices[name].attrs.get("units") != "degC":
            raise ValueError(f"Missing or invalid index {name}; run preprocess.py first.")
    for name in ("ssta", "ssta_relative"):
        if name not in fields or fields[name].dims != ("time", *fields.lat.dims) or fields[name].attrs.get("units") != "degC":
            raise ValueError(f"Missing or invalid anomaly field {name}.")
    observed = np.flatnonzero(np.isfinite(indices[args.event_index].values))
    if not len(observed) or not np.isfinite(indices[args.event_index].values[observed[0]:observed[-1]+1]).all():
        raise ValueError("Event index has no finite values or has interior missing months.")
    LOG.info("Summarizing amplitudes and detecting %s events ...", args.event_index)
    stats = index_statistics(indices)
    events, state = detect_events(indices[args.event_index], args.threshold, args.min_duration)
    event_summary = event_statistics(events, len(observed))
    LOG.info("Computing detrended spectra, autocorrelations, and event composites ...")
    spectrum, autocorrelation, cycles = variability_analysis(indices, args.segment_years, args.max_lag)
    composites = event_composites(fields, events)
    dates = month_labels(indices.time)
    composites.attrs.update(event_index=args.event_index, event_threshold_degC=args.threshold,
                             minimum_duration_months=args.min_duration, analysis_start=dates[0], analysis_end=dates[-1])
    metadata = {
        "source": dict(indices.attrs), "indices_file": str(args.indices.resolve()), "fields_file": str(args.fields.resolve()),
        "analysis_start": dates[0], "analysis_end": dates[-1], "n_months": indices.sizes["time"],
        "event_index": args.event_index, "threshold_degC": args.threshold, "minimum_duration_months": args.min_duration,
        "event_definition": "At least minimum_duration_months consecutive monthly centers of three-month means >= +threshold or <= -threshold",
        "boundary_events": "Flagged as left/right censored; observed duration is a lower bound; included in composites",
        "frequency_denominator": "Number of finite event-index months / 120 decades; descriptive rate, not an occurrence probability",
        "composite_weighting": "Average months within each event, then average events with equal weight",
        "spectrum": {"input": "Unsmoothed nino34 and relative_nino34", "detrend": "Full selected series: linear least squares",
                     "method": "Welch, periodic Hann window, 50% overlap, fs=12 samples/year, density scaling, no additional segment detrending",
                     "requested_segment_years": args.segment_years, "peak_search_years": [1, 20],
                     "enso_band_years": [2, 7], "interpretation": "Discrete-bin peaks and band power; no significance test"},
        "autocorrelation": "Sum of lagged products of detrended data / full-series sum of squares (biased normalization)",
        "std": "Sample standard deviation, ddof=1", "index_caveat": "Fixed-base course indices; RONI-style has no official variance rescaling",
        "purpose": "Descriptive simulated ENSO; not official event classification or observational validation",
    }
    LOG.info("Saving statistics, composite fields, and four figures to %s ...", args.output_dir)
    for frame, name in ((stats, "index_statistics.csv"), (events, "events.csv"), (event_summary, "event_summary.csv"),
                        (spectrum, "spectrum.csv"), (autocorrelation, "autocorrelation.csv"), (cycles, "cycle_summary.csv")):
        write_table(frame, args.output_dir / name)
    write_netcdf(composites, args.output_dir / "enso_composites.nc", args.overwrite)
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    make_figures(indices, events, state, spectrum, autocorrelation, composites,
                 args.output_dir / "figures", args.event_index, args.threshold)
    for row in event_summary.itertuples():
        LOG.info("%s: %s events (%s censored), mean observed duration %.2f months",
                 row.phase, row.n_events, row.n_censored_events, row.mean_observed_duration_months)
    for row in cycles.itertuples():
        LOG.info("%s: largest PSD bin within 1–20 years = %s years", row.index, row.peak_period_1_20_years)
    LOG.info("Finished in %.2f s", perf_counter()-started)


if __name__ == "__main__":
    main()
