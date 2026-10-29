"""
Download and reduce many years of ERA5 - the training record for the V3 LSTM.

For each year: download one GRIB from the Copernicus CDS (same domain and
variables as the 2025 file), reduce it with era5_series.reduce_grib(), and write
data/processed/domain_mean_hourly_<year>.csv. A year whose CSV already has every
column is skipped, so the script can be stopped and re-run - Colab sessions time
out - without redoing finished years.

Needs a free CDS account and its API key in ~/.cdsapirc
(https://cds.climate.copernicus.eu/how-to-api), plus:
    pip install cdsapi xarray cfgrib eccodes

Usage:
    python scripts/era5_years.py                     # 2006-2025, the last 20 years
    python scripts/era5_years.py --years 2015-2020
    python scripts/era5_years.py --years 2026        # current year: complete months only
    python scripts/era5_years.py --raw /content/drive/MyDrive/era5_raw --delete-grib
    python scripts/era5_years.py --dry-run           # print the requests, download nothing

Each year is ~440 MB of GRIB and ~1 MB of CSV. The CDS queues requests, so one
year can take from a few minutes to an hour to come back.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from era5_series import RAW, PROCESSED, GRIB as GRIB_2025, CDS_AREA, reduce_grib

COLUMNS = ["t2m", "d2m", "u10", "v10", "wspd", "wspd_vec", "tp", "t2m_adv"]
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature",
             "10m_u_component_of_wind", "10m_v_component_of_wind",
             "total_precipitation"]
ERA5_LAG_DAYS = 6           # ERA5 is published ~5 days behind real time


def csv_path(year):
    return PROCESSED / f"domain_mean_hourly_{year}.csv"


def is_done(year):
    """True if the year's CSV exists and has every column the V3 model can use."""
    p = csv_path(year)
    if not p.exists():
        return False
    return set(COLUMNS) <= set(pd.read_csv(p, nrows=0).columns)


def months_for(year, today=None):
    """Months of `year` the CDS can serve in full (all 12 for past years)."""
    today = today or dt.date.today()
    latest = today - dt.timedelta(days=ERA5_LAG_DAYS)
    if year < latest.year:
        return list(range(1, 13))
    if year > latest.year:
        return []
    return list(range(1, latest.month))      # the current month is still partial


def request(year, months):
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],   # the CDS skips invalid dates
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": CDS_AREA,
    }


def download(year, months, raw_dir):
    """Fetch one year's GRIB unless it is already on disk. Returns its path."""
    target = raw_dir / f"era5_{year}.grib"
    if target.exists():
        print(f"  {target.name} already downloaded")
        return target
    import cdsapi                                   # only needed to download
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".grib.part")      # never leave a truncated .grib
    print(f"  requesting {year} ({len(months)} months) from the CDS ...", flush=True)
    cdsapi.Client().retrieve("reanalysis-era5-single-levels",
                             request(year, months), str(partial))
    partial.rename(target)
    return target


def process(year, raw_dir, delete_grib=False):
    months = months_for(year)
    if not months:
        print(f"{year}: no complete month available yet - skipped")
        return
    # the original 2025 download keeps its CDS file name; no need to fetch it again
    grib = GRIB_2025 if year == 2025 and GRIB_2025.exists() else \
        download(year, months, raw_dir)

    df = reduce_grib(grib)
    df = df[df.index.year == year]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path(year), float_format="%.6f")
    print(f"{year}: {len(df)} hours, {df.index[0]} .. {df.index[-1]} -> "
          f"{csv_path(year).name}")

    if delete_grib and grib != GRIB_2025:
        grib.unlink()
        for idx in grib.parent.glob(grib.name + ".*.idx"):
            idx.unlink()


def parse_years(text):
    if "-" in text:
        a, b = (int(x) for x in text.split("-"))
        return list(range(a, b + 1))
    return [int(y) for y in text.split(",")]


def main():
    this_year = dt.date.today().year
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--years", default=f"{this_year - 20}-{this_year - 1}",
                   help="a range (2006-2025) or a list (2019,2020); "
                        "default: the 20 complete years before this one")
    p.add_argument("--raw", type=Path, default=RAW,
                   help="where GRIBs are downloaded (default data/raw/)")
    p.add_argument("--delete-grib", action="store_true",
                   help="delete each GRIB once its CSV is written (saves ~440 MB/year)")
    p.add_argument("--force", action="store_true", help="rebuild CSVs that already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be requested and stop")
    args = p.parse_args()

    years = parse_years(args.years)
    todo = [y for y in years if args.force or not is_done(y)]
    print(f"years {years[0]}-{years[-1]}: {len(years) - len(todo)} done, "
          f"{len(todo)} to build")

    if args.dry_run:
        for y in todo:
            print(f"\n{y}:", json.dumps(request(y, months_for(y))))
        return

    for y in todo:
        process(y, args.raw, args.delete_grib)


if __name__ == "__main__":
    main()
