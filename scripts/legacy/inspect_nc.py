"""Inspect the ERA5 NetCDF files: list variables, dimensions, coords, time range."""
import sys
import xarray as xr

files = [
    "382c46a9358c879403b826b3bb39ad87/data_stream-oper_stepType-accum.nc",
    "382c46a9358c879403b826b3bb39ad87/data_stream-oper_stepType-instant.nc",
]

for f in files:
    print("=" * 80)
    print("FILE:", f)
    print("=" * 80)
    try:
        ds = xr.open_dataset(f)
    except Exception as e:
        print("  open failed:", e)
        continue
    print("\nDIMENSIONS:", dict(ds.sizes))
    print("\nCOORDINATES:")
    for c in ds.coords:
        v = ds.coords[c]
        extra = ""
        try:
            extra = f"  [{v.values.min()} .. {v.values.max()}]"
        except Exception:
            pass
        print(f"  {c}: {v.dtype} {v.shape}{extra}")
    print("\nDATA VARIABLES:")
    for name in ds.data_vars:
        v = ds[name]
        long_name = v.attrs.get("long_name", "")
        units = v.attrs.get("units", "")
        print(f"  {name}: dims={v.dims} shape={v.shape}  '{long_name}' [{units}]")
    ds.close()
    print()
