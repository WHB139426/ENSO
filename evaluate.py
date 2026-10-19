"""Evaluate saved SST forecasts on validation and held-out simulation years.

Run: python evaluate.py
This script does not fit or select models.
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

from data import ROOT, Region, month_number, validate_monthly_time, write_netcdf

LOG = logging.getLogger(__name__)
OUTPUTS = ("metrics.csv", "monthly_errors.csv", "spatial_errors.nc", "evaluation_metadata.json",
           "figures/nino34_forecasts.png", "figures/forecast_skill.png",
           "figures/sst_error_maps.png", "figures/sst_forecast_example.png")


def labels(time):
    return [f"{int(y):04d}-{int(m):02d}" for y,m in zip(time.dt.year.values,time.dt.month.values)]


def correlation(prediction, truth):
    if len(truth) < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
        return None
    return float(np.corrcoef(prediction,truth)[0,1])


def validate_forecasts(ds):
    required = {"sst_prediction","sst_truth","nino34_prediction","nino34_truth",
                "relative_nino34_prediction","relative_nino34_truth","tropical_ssta_prediction",
                "tropical_ssta_truth","areacello","model_mask","tos_climatology"}
    if not required.issubset(ds.data_vars):
        raise ValueError(f"Forecast file is missing {sorted(required-set(ds.data_vars))}.")
    if set(map(str,ds.method.values)) != {"persistence","pca_ridge"}:
        raise ValueError("Need both persistence and pca_ridge forecasts.")
    if set(map(str,ds.split.values)) != {"validation","test"}:
        raise ValueError("Need validation and test samples; use model.py's forecasts.nc.")
    spatial = ds.lat.dims
    if ds.sst_prediction.dims != ("method","time",*spatial) or ds.sst_truth.dims != ("time",*spatial):
        raise ValueError("Invalid forecast field dimensions.")
    mask = ds.model_mask.values == 1
    if not mask.any() or not (np.isfinite(ds.areacello.values[mask]) & (ds.areacello.values[mask] > 0)).all():
        raise ValueError("Invalid model ocean mask/areas.")
    if not np.isfinite(ds.sst_prediction.values[:,:,mask]).all() or not np.isfinite(ds.sst_truth.values[:,mask]).all():
        raise ValueError("Forecasts and truth must be finite on the same fixed ocean mask.")
    for name in required-{"areacello","model_mask"}:
        if ds[name].attrs.get("units") != "degC":
            raise ValueError(f"Unexpected units for {name}.")
    target_codes = ds.time.dt.year.values*12 + ds.time.dt.month.values-1
    origin_codes = ds.initialization_time.dt.year.values*12 + ds.initialization_time.dt.month.values-1
    if not np.all(target_codes-origin_codes == int(ds.attrs["lead_months"])):
        raise ValueError("Forecast origin/target months do not match the declared lead.")
    if month_number(ds.attrs["base_end"]) > month_number(ds.attrs["train_end"]):
        raise ValueError("Saved climatology extends beyond the training cutoff.")
    for split,cutoff in (("validation","train_end"),("test","validation_end")):
        selected = ds.split.values == split
        validate_monthly_time(ds.time.isel(time=np.flatnonzero(selected)))
        if np.any(origin_codes[selected] <= month_number(ds.attrs[cutoff])):
            raise ValueError(f"{split} forecast origins overlap the preceding fitting/selection period.")
    # Independently reconstruct the submitted index from the submitted SST.
    inside = Region(190,240,-5,5).mask(ds.lat,ds.lon).values
    area = ds.areacello.values[mask].astype("float64")
    nino = np.where(inside[mask],area,0)
    if nino.sum() <= 0:
        raise ValueError("No Niño3.4 area in the saved model mask.")
    nino /= nino.sum()
    normal = ds.tos_climatology.values[ds.time.dt.month.values-1][:,mask]
    for field,indices in (("sst_prediction","nino34_prediction"),("sst_truth","nino34_truth")):
        values = ds[field].values[...,mask]
        expected = (values-normal).astype("float64") @ nino
        np.testing.assert_allclose(ds[indices],expected,atol=2e-6,rtol=0)
    for suffix in ("prediction","truth"):
        np.testing.assert_allclose(ds["relative_nino34_"+suffix],ds["nino34_"+suffix]-ds["tropical_ssta_"+suffix],
                                   atol=2e-6,rtol=0)


def calculate_metrics(ds):
    validate_forecasts(ds)
    mask = ds.model_mask.values == 1
    area = ds.areacello.values[mask].astype("float64")
    weights = area/area.sum()
    rows, monthly, rmse_maps, bias_maps = [], [], [], []
    dates, origins = labels(ds.time), labels(ds.initialization_time)
    for split in ("validation","test"):
        selection = np.flatnonzero(ds.split.values == split)
        truth = ds.sst_truth.values[selection][:,mask].astype("float64")
        per_method_rmse, per_method_bias = [], []
        for method in ds.method.values:
            pred = ds.sel({"method":method}).sst_prediction.values[selection][:,mask].astype("float64")
            error = pred-truth
            monthly_mse = (error**2) @ weights
            row = {"split":split,"method":str(method),"n_samples":len(selection),
                   "sst_rmse_degC":float(np.sqrt(monthly_mse.mean())),
                   "sst_mae_degC":float((abs(error) @ weights).mean()),
                   "sst_bias_degC":float((error @ weights).mean())}
            for prefix in ("nino34","relative_nino34"):
                p = ds[prefix+"_prediction"].sel({"method":method}).values[selection]
                t = ds[prefix+"_truth"].values[selection]
                row[prefix+"_rmse_degC"] = float(np.sqrt(np.mean((p-t)**2)))
                row[prefix+"_mae_degC"] = float(np.mean(abs(p-t)))
                row[prefix+"_bias_degC"] = float(np.mean(p-t))
                row[prefix+"_correlation"] = correlation(p,t)
            bg_error = ds.tropical_ssta_prediction.sel({"method":method}).values[selection]-ds.tropical_ssta_truth.values[selection]
            row["tropical_ssta_rmse_degC"] = float(np.sqrt(np.mean(bg_error**2)))
            rows.append(row)
            for i,index in enumerate(selection):
                monthly.append({"split":split,"method":str(method),"initialization_month":origins[index],
                                "target_month":dates[index],"sst_rmse_degC":float(np.sqrt(monthly_mse[i])),
                                "nino34_error_degC":float(ds.nino34_prediction.sel({"method":method}).values[index]-ds.nino34_truth.values[index]),
                                "relative_nino34_error_degC":float(ds.relative_nino34_prediction.sel({"method":method}).values[index]-ds.relative_nino34_truth.values[index])})
            rmse, bias = np.full(mask.shape,np.nan,dtype="float32"),np.full(mask.shape,np.nan,dtype="float32")
            rmse[mask] = np.sqrt(np.mean(error**2,axis=0))
            bias[mask] = error.mean(axis=0)
            per_method_rmse.append(rmse)
            per_method_bias.append(bias)
        rmse_maps.append(per_method_rmse)
        bias_maps.append(per_method_bias)
    metrics = pd.DataFrame(rows)
    for metric in ("sst","nino34","relative_nino34"):
        column = metric+"_mse_skill_vs_persistence"
        metrics[column] = np.nan
        for split in ("validation","test"):
            selected = metrics.split == split
            baseline = float(metrics.loc[selected & (metrics.method == "persistence"),metric+"_rmse_degC"].iloc[0])
            if baseline > 0:
                metrics.loc[selected,column] = 1-(metrics.loc[selected,metric+"_rmse_degC"]/baseline)**2
    spatial = ds.lat.dims
    coords = {"split":["validation","test"],"method":ds.method.values,"lat":ds.lat,"lon":ds.lon}
    coords.update({dim:ds[dim] for dim in spatial if dim in ds.coords})
    fields = xr.Dataset({"sst_rmse":(("split","method",*spatial),np.array(rmse_maps)),
                         "sst_bias":(("split","method",*spatial),np.array(bias_maps)),
                         "areacello":ds.areacello,"model_mask":ds.model_mask},coords=coords,attrs=dict(ds.attrs))
    fields.sst_rmse.attrs["units"] = fields.sst_bias.attrs["units"] = "degC"
    return metrics,pd.DataFrame(monthly),fields


def make_figures(ds, metrics, spatial_errors, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output.mkdir(parents=True,exist_ok=True)
    test = ds.isel(time=np.flatnonzero(ds.split.values == "test"))
    years = test.time.dt.year.values+(test.time.dt.month.values-0.5)/12
    colors = {"persistence":"tab:blue","pca_ridge":"tab:orange"}
    fig,axes = plt.subplots(2,1,figsize=(12,6),sharex=True,layout="constrained")
    for ax,name in zip(axes,("nino34","relative_nino34")):
        ax.plot(years,test[name+"_truth"],color="black",lw=1.1,label="Truth")
        for method,color in colors.items():
            ax.plot(years,test[name+"_prediction"].sel({"method":method}),color=color,lw=0.8,label=method)
        ax.set(ylabel="Anomaly (°C)",title=f"{name}: {ds.attrs['lead_months']}-month lead, held-out years")
        ax.axhline(0,color="gray",lw=0.5)
        ax.legend(ncol=3)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Target model year")
    fig.savefig(output/"nino34_forecasts.png",dpi=180)
    plt.close(fig)

    fig,axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
    summary = metrics.loc[metrics.split == "test"].set_index("method")
    for ax,(name,title) in zip(axes,(("sst_rmse_degC","SST RMSE (°C); lower is better"),
                                    ("nino34_rmse_degC","Niño3.4 RMSE (°C); lower is better"),
                                    ("nino34_correlation","Niño3.4 correlation; higher is better"))):
        ax.bar(list(colors),[summary.loc[m,name] for m in colors],color=list(colors.values()))
        ax.set_title(title,fontsize=10)
        ax.tick_params(axis="x",rotation=15)
    fig.suptitle("Held-out simulation skill on identical forecast months")
    fig.savefig(output/"forecast_skill.png",dpi=180)
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(12,6),layout="constrained")
    fields = spatial_errors.sel(split="test")
    bias_limit = max(0.1,float(abs(fields.sst_bias).max()))
    rmse_limit = max(0.1,float(fields.sst_rmse.max()))
    for row,method in enumerate(colors):
        for col,name in enumerate(("sst_bias","sst_rmse")):
            ax = axes[row,col]
            image = ax.pcolormesh(fields.lon,fields.lat,fields[name].sel({"method":method}),shading="auto",
                                  cmap="RdBu_r" if col == 0 else "viridis",vmin=-bias_limit if col == 0 else 0,
                                  vmax=bias_limit if col == 0 else rmse_limit)
            ax.add_patch(Rectangle((190,-5),50,10,fill=False,edgecolor="black",lw=1))
            ax.set(title=f"{method}: {name}",xlabel="Longitude (°E)",ylabel="Latitude (°N)")
            fig.colorbar(image,ax=ax,label="°C")
    fig.suptitle("Held-out SST errors; identical color scales within each column")
    fig.savefig(output/"sst_error_maps.png",dpi=180)
    plt.close(fig)

    last = test.isel(time=-1)
    normal = last.tos_climatology.sel(month=int(last.time.dt.month))
    maps = [last.sst_truth-normal,*[last.sst_prediction.sel({"method":m})-normal for m in colors]]
    limit = max(1.,max(float(abs(field).max()) for field in maps))
    fig,axes = plt.subplots(1,3,figsize=(14,4),layout="constrained")
    for ax,field,title in zip(axes,maps,["Truth",*colors]):
        image = ax.pcolormesh(last.lon,last.lat,field,shading="auto",cmap="RdBu_r",vmin=-limit,vmax=limit)
        ax.add_patch(Rectangle((190,-5),50,10,fill=False,edgecolor="black",lw=1))
        ax.set(title=title,xlabel="Longitude (°E)",ylabel="Latitude (°N)")
    fig.colorbar(image,ax=axes.tolist(),label="SST anomaly (°C)",shrink=0.8)
    fig.suptitle(f"Final held-out target month: {labels(test.time)[-1]}; lead = {ds.attrs['lead_months']} months")
    fig.savefig(output/"sst_forecast_example.png",dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"data/model/forecasts.nc")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data/evaluation")
    parser.add_argument("--overwrite",action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    if not args.overwrite and any((args.output_dir/name).exists() for name in OUTPUTS):
        parser.error("Evaluation outputs already exist; use --overwrite or another --output-dir")
    started = perf_counter()
    LOG.info("Loading forecasts and checking lead times, splits, and SST-derived indices ...")
    with xr.open_dataset(args.input,chunks=None) as source:
        ds = source.load()
    metrics,monthly,spatial = calculate_metrics(ds)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    LOG.info("Saving metrics, spatial errors, and four figures ...")
    with tempfile.TemporaryDirectory(prefix=".evaluation-",dir=args.output_dir) as staging:
        staging = Path(staging)
        metrics.to_csv(staging/"metrics.csv",index=False,float_format="%.10g")
        monthly.to_csv(staging/"monthly_errors.csv",index=False,float_format="%.10g")
        write_netcdf(spatial,staging/"spatial_errors.nc")
        # NetCDF decodes integer attributes as NumPy scalars, unlike JSON.
        forecast_metadata = {name: value.item() if isinstance(value,np.generic) else value
                             for name,value in ds.attrs.items()}
        metadata = {"forecast_file":str(args.input.resolve()),"forecast_metadata":forecast_metadata,
                    "field_metrics":"All retained model_mask cells; true grid-cell areas; equal weight per target month",
                    "rmse":"sqrt(mean_over_time(area_weighted_mean_over_cells(error**2)))",
                    "correlation":"Pearson correlation across target months; undefined for constant series",
                    "skill":"1 - model_MSE / persistence_MSE; positive is improvement; undefined for zero baseline MSE",
                    "validation":"Used for model selection; not an independent final score",
                    "test":"Held out from all parameter selection; this script does not tune models",
                    "indices":"Monthly indices from saved SST, using the training climatology and identical model_mask",
                    "example_map":"Final test target month; chosen without searching for favorable forecast skill"}
        (staging/"evaluation_metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
        make_figures(ds,metrics,spatial,staging/"figures")
        for name in OUTPUTS:
            destination = args.output_dir/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            (staging/name).replace(destination)
    for row in metrics.loc[metrics.split == "test"].itertuples():
        LOG.info("Test %s: SST RMSE=%.4f °C, Niño3.4 RMSE=%.4f °C, r=%s, SST MSE skill=%+.3f",
                 row.method,row.sst_rmse_degC,row.nino34_rmse_degC,row.nino34_correlation,row.sst_mse_skill_vs_persistence)
    LOG.info("Saved %s; finished in %.2f s",args.output_dir,perf_counter()-started)


if __name__ == "__main__":
    main()
