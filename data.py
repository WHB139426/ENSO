"""Fetch and cache IPSL-CM6A-LR SST on its native curvilinear ocean grid.

Example: python data.py --start 1985-01 --end 2014-12
No installation, network access, or computation happens on import.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import tempfile

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
CATALOG = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
MODEL = "IPSL-CM6A-LR"
MEMBER = "r1i1p1f1"
EXPERIMENTS = ("historical", "ssp126", "ssp245", "ssp370", "ssp585")
DROP_BOUNDS = [
    "time_bnds", "time_bounds", "lat_bnds", "lon_bnds", "bounds_lat",
    "bounds_lon", "bounds_nav_lat", "bounds_nav_lon",
]


@dataclass(frozen=True)
class Region:
    """Cell-center selection, with longitudes in degrees east; wrapping is allowed."""

    lon_min: float = 120.0
    lon_max: float = 290.0
    lat_min: float = -20.0
    lat_max: float = 20.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("Region bounds must be finite.")
        if not -90 <= self.lat_min < self.lat_max <= 90:
            raise ValueError("Latitude bounds must satisfy -90 <= min < max <= 90.")
        if not 0 < abs(self.lon_max - self.lon_min) < 360:
            raise ValueError("Longitude span must be nonzero and less than 360 degrees.")

    def mask(self, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
        lo, hi = self.lon_min % 360, self.lon_max % 360
        longitude = lon % 360
        inside_lon = ((longitude >= lo) & (longitude <= hi) if lo < hi
                      else (longitude >= lo) | (longitude <= hi))
        return inside_lon & (lat >= self.lat_min) & (lat <= self.lat_max)


def month_number(value: str) -> int:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError(f"Expected YYYY-MM, got {value!r}.")
    year, month = map(int, value.split("-"))
    if year < 1 or not 1 <= month <= 12:
        raise ValueError(f"Invalid month: {value!r}.")
    return year * 12 + month - 1


def validate_monthly_time(time: xr.DataArray) -> np.ndarray:
    """Require exactly one record per consecutive month, retaining CF calendars."""
    if time.dims != ("time",) or time.size == 0:
        raise ValueError("Expected a nonempty one-dimensional time coordinate.")
    try:
        years, months = time.dt.year.values, time.dt.month.values
        if not (np.isfinite(years).all() and np.isfinite(months).all()):
            raise ValueError("Time contains missing dates.")
        codes = years.astype(np.int64) * 12 + months.astype(np.int64) - 1
    except (AttributeError, TypeError) as exc:
        raise ValueError("Time must contain decoded datetime or cftime dates.") from exc
    if np.any(np.diff(codes) != 1):
        raise ValueError("Time has duplicate, missing, or out-of-order months.")
    return codes


def select_months(obj, start: str | None, end: str | None):
    if start and end and month_number(start) > month_number(end):
        raise ValueError("Start month must not be after end month.")
    selected = obj.sel(time=slice(start, end))
    codes = validate_monthly_time(selected.time)
    if start and codes[0] != month_number(start):
        raise ValueError(f"Requested start month {start} is not available.")
    if end and codes[-1] != month_number(end):
        raise ValueError(f"Requested end month {end} is not available.")
    return selected


def normalize_sst(sst: xr.DataArray) -> xr.DataArray:
    """Validate native-grid geometry and convert supported temperature units to Celsius."""
    for coord in ("time", "lat", "lon"):
        if coord not in sst.coords:
            raise ValueError(f"SST is missing coordinate {coord!r}.")
    if sst.lat.ndim != 2 or sst.lat.dims != sst.lon.dims:
        raise ValueError("Expected two-dimensional lat/lon on the same native grid.")
    spatial = sst.lat.dims
    for dim in set(sst.dims) - {"time", *spatial}:
        if sst.sizes[dim] != 1:
            raise ValueError(f"Unexpected non-singleton SST dimension {dim!r}.")
        sst = sst.isel({dim: 0}, drop=True)
    if set(sst.dims) != {"time", *spatial}:
        raise ValueError("SST must have time and exactly two spatial dimensions.")
    sst = sst.transpose("time", *spatial)
    validate_monthly_time(sst.time)
    original = str(sst.attrs.get("units", ""))
    unit = original.lower().replace(" ", "").replace("°", "")
    if unit in {"k", "kelvin", "degree_k", "degrees_k"}:
        sst = sst - 273.15
    elif unit not in {"c", "degc", "celsius", "degree_celsius", "degrees_celsius",
                      "degree_c", "degrees_c"}:
        raise ValueError(f"Unknown SST units {original!r}; specify valid input metadata.")
    # Mask BEFORE changing longitude coordinates. Otherwise xarray can drop lon
    # while reconciling a condition carrying the old -180..180 coordinates.
    sst = sst.where(np.isfinite(sst))
    sst = sst.assign_coords(lon=sst.lon % 360)
    sst.name = "tos"
    sst.attrs = {"units": "degC", "long_name": "Sea surface temperature",
                 "original_units": original}
    sst.lat.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    sst.lon.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return sst


def align_area(sst: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    """Reject index/geometry mismatches rather than silently reindexing cell areas."""
    spatial = sst.lat.dims
    for dim in set(area.dims) - set(spatial):
        if area.sizes[dim] != 1:
            raise ValueError(f"Unexpected area dimension {dim!r}.")
        area = area.isel({dim: 0}, drop=True)
    if set(area.dims) != set(spatial):
        raise ValueError(f"Area dimensions {area.dims} do not match SST {spatial}.")
    area = area.transpose(*spatial)
    for dim in spatial:
        if area.sizes[dim] != sst.sizes[dim]:
            raise ValueError(f"Area/SST size mismatch on {dim}.")
        if (dim in area.coords) != (dim in sst.coords):
            raise ValueError(f"Only one input has grid indices for {dim}.")
        if dim in area.coords and not area[dim].equals(sst[dim]):
            raise ValueError(f"Area/SST grid indices differ on {dim}.")
    for name in ("lat", "lon"):
        if name not in area.coords:
            raise ValueError(f"Area needs physical {name} coordinates to verify the grid.")
        other = area[name].transpose(*spatial)
        difference = other - sst[name]
        if name == "lon":
            difference = (difference + 180) % 360 - 180
        if not bool((np.isfinite(difference) & (abs(difference) < 1e-5)).all().compute()):
            raise ValueError(f"Area/SST physical {name} coordinates differ.")
    unit = str(area.attrs.get("units", "")).lower().replace(" ", "")
    if unit not in {"m2", "m^2", "m**2", "m²"}:
        raise ValueError(f"Area must be in square meters, got {unit!r}.")
    # Reattach the verified SST coordinates, avoiding conflicts on longitude conventions.
    return xr.DataArray(area.data, dims=spatial,
                        coords={d: sst[d] for d in spatial if d in sst.coords},
                        name="areacello", attrs={"units": "m2"})


def area_mean(field: xr.DataArray, area: xr.DataArray,
              mask: xr.DataArray) -> xr.DataArray:
    """Area mean with the denominator recomputed over each month's valid cells."""
    spatial = area.dims
    if not set(spatial).issubset(field.dims) or set(mask.dims) != set(spatial):
        raise ValueError("Field, area, and region mask must share spatial dimensions.")
    field, area, mask = xr.align(field, area, mask, join="exact")
    weights = area.where(mask & np.isfinite(area) & (area > 0), 0).fillna(0)
    if not bool((weights.sum() > 0).compute()):
        raise ValueError("Selected region contains no positive-area cells.")
    return field.where(np.isfinite(field)).weighted(weights).mean(spatial, skipna=True)


def prepare_data(sst: xr.DataArray, area: xr.DataArray,
                 region: Region = Region(), start: str | None = None,
                 end: str | None = None) -> xr.Dataset:
    """Produce regional SST plus a GLOBAL 20S-20N ocean mean before cropping."""
    sst = select_months(normalize_sst(sst), start, end)
    area = align_area(sst, area)
    tropical_mask = (sst.lat >= -20) & (sst.lat <= 20)
    tropical = area_mean(sst, area, tropical_mask).rename("tropical_mean_sst")
    tropical.attrs = {"units": "degC", "long_name": "Global tropical ocean mean SST",
                      "region": "20S-20N, all longitudes; cell-center selection"}
    inside = region.mask(sst.lat, sst.lon)
    valid = inside & np.isfinite(area) & (area > 0)
    valid_values = valid.compute().values
    if not valid_values.any():
        raise ValueError("Pacific region contains no positive-area cells.")
    slices = {}
    for axis, dim in enumerate(inside.dims):
        indices = np.flatnonzero(valid_values.any(axis=1 - axis))
        slices[dim] = slice(int(indices[0]), int(indices[-1]) + 1)
    crop = inside.isel(slices)
    ds = xr.Dataset({
        "tos": sst.isel(slices).where(valid.isel(slices)),
        "areacello": area.isel(slices),
        "pacific_mask": crop.astype("int8"),
        "tropical_mean_sst": tropical,
    })
    ds.pacific_mask.attrs = {"long_name": "Requested Pacific region (1=inside, 0=outside)"}
    ds.attrs = {
        "region_bounds": json.dumps(vars(region)),
        "tropical_bounds": "20S-20N, all longitudes",
        "spatial_selection": "Grid cell centers; no regridding or fractional-cell weighting",
        "missing_data": "Each mean renormalizes weights over finite SST and positive area",
    }
    return ds


def _select_store(frame, label: str):
    if frame.empty:
        raise ValueError(f"No cloud catalog entry for {label}.")
    frame = frame.drop_duplicates("zstore")
    if "version" in frame and len(frame) > 1:
        frame = frame.loc[frame.version == frame.version.max()]
    if len(frame) != 1:
        raise ValueError(f"Ambiguous cloud catalog selection for {label}: {len(frame)} stores.")
    return frame.iloc[0]


def open_cmip6(experiment: str = "historical", model: str = MODEL,
               member: str = MEMBER, catalog_url: str = CATALOG):
    """Find one SST store and a same-grid area store; return lazy DataArrays and provenance."""
    try:
        import intake
        import intake_esm  # noqa: F401; registers the intake catalog opener
        from xmip.preprocessing import combined_preprocessing
    except ImportError as exc:
        raise ImportError("Cloud access needs intake-esm, xmip, gcsfs and zarr; see requirements.txt.") from exc
    LOG.info("Querying cloud catalog for %s/%s/%s", model, member, experiment)
    catalog = intake.open_esm_datastore(catalog_url)
    rows = catalog.search(source_id=model, member_id=member, experiment_id=experiment,
                          variable_id="tos", table_id="Omon", grid_label="gn").df
    tos_row = _select_store(rows, "tos")
    areas = catalog.search(source_id=model, variable_id="areacello", table_id="Ofx",
                           grid_label="gn").df
    if "institution_id" in areas:
        areas = areas.loc[areas.institution_id == tos_row.institution_id]
    # Cell areas are static, so another member/experiment is acceptable ONLY after
    # physical grid coordinates have been verified by align_area().
    for column, preferred in (("member_id", member), ("experiment_id", experiment),
                              ("experiment_id", "historical")):
        matches = areas.loc[areas[column] == preferred]
        if not matches.empty:
            areas = matches
    area_row = _select_store(areas, "areacello")

    def open_row(row, variable):
        import zarr
        # Own each filesystem so closing one array cannot invalidate another.
        # Zarr 3 does I/O on its own event loop; its session must close there too,
        # rather than in gcsfs's interpreter-shutdown finalizer on another loop.
        if int(zarr.__version__.split(".")[0]) >= 3:
            from zarr.storage import FsspecStore
            from zarr.core.sync import sync
            location = FsspecStore.from_url(
                str(row.zstore), read_only=True,
                storage_options={"token": "anon", "skip_instance_cache": True})
            filesystem = location.fs

            def close_session(session):
                sync(session.close())
        else:
            import gcsfs
            from fsspec.asyn import sync
            filesystem = gcsfs.GCSFileSystem(token="anon", skip_instance_cache=True)
            location = filesystem.get_mapper(str(row.zstore))

            def close_session(session):
                sync(filesystem.loop, session.close)

        raw = None

        def close():
            if raw is not None:
                raw.close()
            try:
                session = filesystem.session
            except RuntimeError:  # No connection was opened.
                return
            if session is not None and not session.closed:
                close_session(session)

        try:
            raw = xr.open_zarr(location, consolidated=True, chunks={}, drop_variables=DROP_BOUNDS)
            normalized = combined_preprocessing(raw)
            result = normalized[variable]
            result.set_close(close)
            return result
        except Exception:
            close()
            raise

    sst = open_row(tos_row, "tos")
    try:
        area = open_row(area_row, "areacello")
    except Exception:
        sst.close()
        raise
    provenance = {"source_id": model, "member_id": member, "experiment_id": experiment,
                  "grid_label": "gn", "catalog_url": catalog_url,
                  "tos_zstore": str(tos_row.zstore), "area_zstore": str(area_row.zstore)}
    return sst, area, provenance


def write_netcdf(ds: xr.Dataset, path: str | Path, overwrite: bool = False) -> Path:
    """Compressed NetCDF written via a temporary file, preserving an existing cache on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; use --overwrite to replace it.")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".nc", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        clean = ds.drop_encoding()
        encoding = {name: {"zlib": True, "complevel": 4}
                    for name, var in clean.data_vars.items() if var.ndim}
        clean.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--member", default=MEMBER)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="historical")
    parser.add_argument("--start", help="First month YYYY-MM; default 1850-01 (SSP: 2015-01)")
    parser.add_argument("--end", help="Last month YYYY-MM; default 2014-12 (SSP: 2100-12)")
    parser.add_argument("--lon-min", type=float, default=120)
    parser.add_argument("--lon-max", type=float, default=290)
    parser.add_argument("--lat-min", type=float, default=-20)
    parser.add_argument("--lat-max", type=float, default=20)
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--output", type=Path, help="Cache path; default data/raw/MODEL_EXPERIMENT.nc")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    historical = args.experiment == "historical"
    start = args.start or ("1850-01" if historical else "2015-01")
    end = args.end or ("2014-12" if historical else "2100-12")
    if month_number(start) > month_number(end):
        parser.error("--start must not be after --end")
    region = Region(args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    output = args.output or ROOT / "data" / "raw" / f"{args.model}_{args.experiment}.nc"
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; use --overwrite or another --output")
    sst, area, provenance = open_cmip6(args.experiment, args.model, args.member, args.catalog)
    try:
        ds = prepare_data(sst, area, region, start, end)
        ds.attrs.update(provenance, requested_start=start, requested_end=end)
        LOG.info("Selected %s months; regional grid %s. Computing and saving %s ...",
                 ds.sizes["time"], dict(ds.tos.sizes), output)
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            write_netcdf(ds, output, args.overwrite)
    finally:
        sst.close()
        area.close()
    LOG.info("Saved %s", output)


if __name__ == "__main__":
    main()
