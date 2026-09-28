#!/usr/bin/env python3

import argparse

import pandas as pd
import numpy as np
import os
import warnings
from datetime import timedelta
import xarray as xr
import glob
warnings.filterwarnings('ignore')

# Find days where ALL three datasets have data, then mask to those days only
def get_valid_days(ds, var='rain_intensity'):
    """Return a set of dates (day resolution) where the variable has at least one non-NaN value."""
    data = ds[var] if var in ds else next(iter(ds.data_vars.values()), None)
    if data is None:
        return set()
    # Collapse any non-time dims, check for any valid value per day
    daily_valid = (
        np.isfinite(data.values).any(axis=tuple(range(1, data.ndim)))
        if data.ndim > 1 else np.isfinite(data.values)
    )
    times = ds.time.values
    days = pd.to_datetime(times).normalize()  # truncate to date
    return set(days[daily_valid])

# Build a boolean mask for each dataset's time coordinate
def mask_to_common_days(ds, common_days):
    common_days = pd.DatetimeIndex(common_days).normalize()
    time_days = ds["time"].dt.floor("D")
    mask = time_days.isin(common_days)
    ds_filtered = ds.where(mask, drop=True)
    return ds_filtered

def main(year, month):
    # Define datasets to load: (name, glob_pattern, variable, height_level, y_offset, color)
    datasets_config = [
        ('Asit Winds', f'/qfs/people/gold396/ORACLE/wfip3/asit.lidar.z01.a0/asit.lidar.z01.a0.{year}{month}*', 'horizontal_wind_direction', 97, 0.0, 'orange'),
        ('Asit Rain 1', f'/rcfs/projects/oracle/gold396/Disdrometer_Data/wfip3/asit.ld.z01.b1/asit.ld.z01.b1.{year}{month}*', 'rain_intensity', None, -0.25, 'orange'),
        ('Asit Rain 2', f'/rcfs/projects/oracle/gold396/Disdrometer_Data/wfip3/asit.ld.z02.b1/asit.ld.z02.b1.{year}{month}*', 'rain_intensity', None, -0.75, 'orange'),
    ]

    # Load all datasets efficiently
    datasets = {}
    for name, pattern, _, _, _, _ in datasets_config:
        try:
            ds = xr.open_mfdataset(sorted(glob.glob(pattern)), combine='by_coords')
            datasets[name] = ds
            print(f"✓ Loaded {name}: {len(ds.time)} time steps")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")




    wind_days = get_valid_days(datasets['Asit Winds'], 'horizontal_wind_direction')
    rain1_days = get_valid_days(datasets['Asit Rain 1'], 'rain_intensity')
    rain2_days = get_valid_days(datasets['Asit Rain 2'], 'rain_intensity')

    common_days = wind_days & rain1_days & rain2_days
    print(f"Asit Winds valid days:  {len(wind_days)}")
    print(f"Asit Rain 1 valid days: {len(rain1_days)}")
    print(f"Asit Rain 2 valid days: {len(rain2_days)}")
    print(f"Overlapping days:       {len(common_days)}")

    datasets['Asit Winds']  = mask_to_common_days(datasets['Asit Winds'],  common_days)
    datasets['Asit Rain 1'] = mask_to_common_days(datasets['Asit Rain 1'], common_days)
    datasets['Asit Rain 2'] = mask_to_common_days(datasets['Asit Rain 2'], common_days)

    print(f"\nAfter masking:")
    for name in ['Asit Winds', 'Asit Rain 1', 'Asit Rain 2']:
        print(f"  {name}: {len(datasets[name].time)} time steps")

    del wind_days, rain1_days, rain2_days, common_days

    # Make sure not time duplicates
    datasets["Asit Rain 1"] = datasets["Asit Rain 1"].sortby("time")
    datasets["Asit Rain 2"] = datasets["Asit Rain 2"].sortby("time")
    datasets["Asit Winds"] = datasets["Asit Winds"].sortby("time")
    datasets["Asit Rain 1"] = datasets["Asit Rain 1"].drop_duplicates(dim="time", keep="last")
    datasets["Asit Rain 2"] = datasets["Asit Rain 2"].drop_duplicates(dim="time", keep="last")
    datasets["Asit Winds"] = datasets["Asit Winds"].drop_duplicates(dim="time", keep="last")

    # Make sure eveything has same time coordinate
    datasets['Asit Winds'] = datasets['Asit Winds'].reindex(time=datasets["Asit Rain 1"]["time"], method="ffill")
    datasets['Asit Rain 2'] = datasets['Asit Rain 2'].reindex(time=datasets["Asit Rain 1"]["time"], method="ffill")

    WD = datasets['Asit Winds']['horizontal_wind_direction'].sel(height=100, method = 'nearest').compute()

    ds1 = datasets["Asit Rain 1"]   # Dataset
    ds2 = datasets["Asit Rain 2"]   # Dataset

    # Boolean masks (True where condition met)
    mask_ns = ((WD > 315) | (WD < 45) | ((WD > 135) & (WD < 225)))
    mask_ew = (((WD > 45) & (WD < 135)) | ((WD > 225) & (WD < 315)))

    new_ds = xr.where(mask_ns, ds1, xr.where(mask_ew, ds2, np.nan))
    new_ds["rain_intensity_ns"] = ds1["rain_intensity"]
    new_ds["rain_intensity_ew"] = ds2["rain_intensity"]
    new_ds["wind_direction"] = WD
    new_ds["wind_speed"] = datasets["Asit Winds"]["horizontal_wind_speed"].sel(height=100, method='nearest').compute()
    new_ds["vertical_wind_speed"] = datasets["Asit Winds"]["vertical_wind_speed"].sel(height=100, method='nearest').compute()

    output_dir = '/rcfs/projects/oracle/gold396/ASIT_submit2'
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted unique dates from new_ds
    dates = sorted(set(pd.to_datetime(new_ds.time.values).normalize()))

    for date in dates:
        date_str = pd.Timestamp(date).strftime('%Y%m%d')
        
        # Extract all data for this day from new_ds
        day_ds = new_ds.sel(time=slice(date, date + timedelta(days=1) - timedelta(seconds=1)))
        
        # Compute to materialize Dask arrays before saving
        day_ds = day_ds.compute()
        
        # Save to daily file
        output_file = os.path.join(output_dir, f"asit_merged_{date_str}.nc")
        day_ds.to_netcdf(output_file)
        print(f"✓ Saved {date_str}: {output_file}")

    print(f"\nTotal daily files saved: {len(dates)}")


    # The entire script content should be indented under this function
    # (from line 11 to line 112)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    args = parser.parse_args()

    main(args.year, args.month)
