#{1. Filter if raining R > 0.1 mm/h
# 2. Reduce to 1 minute
# 3. Flag two smallest size bins
# 4. Flag if exceeds +/- 50% terminal velocity
# 5. Fit distribution, if outside shape factor -2 to 20, flag
# 6. Add flags to original dataset}
#}

import argparse
import dask
import pandas as pd
import numpy as np
import os
from io import BytesIO
import warnings
from datetime import datetime, timedelta
import xarray as xr
import glob

def correct_file(in_path, new_filename):

    # 1. LOAD FILE
    with xr.open_dataset(in_path) as ds:
        

        # 2. FILTER TO PERIODS OF INTEREST
        condition = ds["rain_intensity"].fillna(0) > 0.1
        keep_time = condition.compute()

        ds_filtered = (
            ds.where(condition)
            .isel(time=keep_time)
        )

        if ds_filtered.sizes.get("time", 0) == 0:
            raise ValueError(f"No rainy timestamps in {in_path}")

        # 3. FILTER OUT SMALLEST BINS

        mask = (ds_filtered["diameter_ve"] >= 0.2) 
        ds_filtered = ds_filtered.isel(diameter_ve=mask)

        # 4. REDUCE TO 1 MINUTE

        sum_vars = ['n_particles',
        'dve_bin_concentration',
        'fs_bin_concentration',
        'dve_fs_bin_concentration'
        ]
        mean_vars = [ 'rain_intensity',
        'wind_speed',
        'vertical_wind_speed',
        'radar_reflectivity',
        'visibility',
        ]
        first_vars = ['wind_direction',
        'weather_code_nws',
        ]
        max_vars = ['qc_rain_intensity',
        'qc_weather_code_nws',
        'qc_radar_reflectivity',
        'qc_visibility',
        'qc_n_particles',
        'qc_dve_bin_concentration',
        'qc_fs_bin_concentration',
        'qc_dve_fs_bin_concentration'
        ]

        result = xr.merge([
            ds_filtered[sum_vars]
                .resample(time="1min")
                .sum(min_count=1),

            ds_filtered[mean_vars]
                    .resample(time="1min")
                    .mean(),

            ds_filtered[first_vars]
                .resample(time="1min")
                .first(skipna=True),

            ds_filtered[max_vars]
                .resample(time="1min")
                .max(skipna=True),
            
        ])

        # 5. FILTER BY TERMINAL VELOCITY
        da = result["dve_fs_bin_concentration"]

        # Broadcast data and coordinates to identical dimensions
        data, diameter, fall_speed = xr.broadcast(
            da,
            da["diameter_ve"],
            da["fall_speed_ms"],
        )

        # Terminal-velocity curve and its boundaries
        vt = 9.65 - 10.3 * np.exp(-0.6 * diameter)

        line_low = np.minimum(vt * 0.5, vt * 1.5)
        line_high = np.maximum(vt * 0.5, vt * 1.5)

        # True only for nonzero points inside the boundaries
        inside = (
            data.notnull()
            & (data != 0)
            & (fall_speed >= line_low)
            & (fall_speed <= line_high)
        )

        # Outside points become NaN
        filtered = data.where(inside)

        result["dve_fs_bin_concentration_filtered"] = filtered

        # 6. FILTER BY SHAPE PARAMETER
        da = result['dve_bin_concentration']
        db = result['dve_fs_bin_concentration_filtered']

        SD = []
        SDD = []

        for i in range(len(da.time.values)):
            t = da.time.values[i]
            single_distribution = da.sel(time=t)
            single_2d_distribution = db.sel(time=t)


            # 1D x and y arrays
            x = single_distribution["diameter_ve"]
            y = single_distribution

            x_values = x.values
            y_values = y.values

            # Boundary equations
            a = 2
            b = 0.4

            y_max = y.max(skipna=True).values

            lower_line = (y_max / a) * np.exp(-a * x_values)
            upper_line = (y_max / b) * np.exp(-b * x_values)

            line_low = np.minimum(lower_line, upper_line)
            line_high = np.maximum(lower_line, upper_line)

            # Valid, nonzero data
            valid = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & (y_values != 0)
            )

            inside = valid & (y_values >= line_low) & (y_values <= line_high)
            inside_x = xr.DataArray(inside,dims='diameter_ve',coords={'diameter_ve': da['diameter_ve']})


            SD.append(single_distribution.where(inside))
            SDD.append(single_2d_distribution.where(inside_x))

        result['dve_bin_concentration_filtered'] = xr.concat(SD, dim="time")
        result['dve_fs_bin_concentration_filtered2'] = xr.concat(SDD, dim="time")

        # 7. CALCULATE NEW RAIN RATE
        da = result['dve_fs_bin_concentration_filtered2']

        # Keep only non-NaN concentration values
        valid_da = da.where(da.notnull(), 0)

        # Volume for each diameter
        Vj = (np.pi/6) * ((da['diameter_ve'] / 2)*0.001) ** 3
        # Calculate R; xarray broadcasts Vj and fall speed across the 2D grid
        R = valid_da * Vj * da['fall_speed_ms'] 

        # Sum over diameter and fall-speed for each time
        RR = R.sum(
            dim=['diameter_ve', 'fall_speed_ms'],
            skipna=True
        )* ((1000*3600) / (60*0.0054))

        result['rain_intensity_corrected'] = RR.rename('RR')
        result['rain_intensity_corrected'].attrs["methods"] = "calculated from dve_fs_bin_concentration_filtered2 following the standard rain rate formula"
        result['rain_intensity_corrected'].attrs["units"] = "mm/h"

        # 8. Calculate new visibility

        da = result['dve_bin_concentration_filtered']

        # Keep only non-NaN concentration values
        valid_da = da.where(da.notnull(), 0)

        # Calculate extinction coefficient
        sigma = (np.pi/2) * ((da['diameter_ve'])) ** 2 * valid_da


        # Sum over diameter and MOR for each time
        MOR = 20000-(sigma.sum(
            dim=['diameter_ve'],
            skipna=True
        ))*10

        result['visibility_corrected'] = MOR.rename('MOR')
        result['visibility_corrected'].attrs["methods"] = "calculated from dve_bin_concentration_filtered following Koschmieders equation and the atmospheric extinction coefficient simplified assuming mie scattering theory"
        result["visibility_corrected"].attrs["units"] = "m"

        # 8. Calculate new radar reflectivity
        da = result['dve_bin_concentration_filtered']

        # Keep only non-NaN concentration values
        valid_da = da.where(da.notnull(), 0)

        # Calculate linear radar reflectivity factor (Z)
        Z = (((1/6)*np.pi*(da['diameter_ve']) ** 3)**-1)*((da['diameter_ve']) ** 6) * valid_da

        # Sum over diameter and calculate dbz for each time
        zsum = Z.sum(dim="diameter_ve", skipna=True)
        dbz = 10 * np.log10(zsum.where(zsum > 0))

        result['radar_reflectivity_corrected'] = dbz.rename('dbz')
        result['radar_reflectivity_corrected'].attrs["methods"] = "calculated from dve_bin_concentration_filtered following the standard radar reflectivity formula"
        result['radar_reflectivity_corrected'].attrs["units"] = "dBZ"
        # 9. SAVE NEW FILE

        # RR_on_ds_time = RR.reindex(
        #     time=ds.time,
        #     method='nearest',
        #     tolerance=np.timedelta64(30, 's')  # adjust as appropriate
        # )

        # ds['rain_intensity_corrected'] = RR_on_ds_time

        # RR_on_ds_time = result['dve_fs_bin_concentration_filtered2'].reindex(
        #     time=ds.time,
        #     method='nearest',
        #     tolerance=np.timedelta64(30, 's')  # adjust as appropriate
        # )

        # ds['dve_fs_bin_concentration_corrected'] = RR_on_ds_time

        # RR_on_ds_time = result['dve_bin_concentration_filtered'].reindex(
        #     time=ds.time,
        #     method='nearest',
        #     tolerance=np.timedelta64(30, 's')  # adjust as appropriate
        # )

        # ds['dve_bin_concentration_corrected'] = RR_on_ds_time

        # ds["dve_bin_concentration_corrected"].attrs["methods"] = "filtered by shape parameter following Tokay et al 2013"
        # ds["dve_fs_bin_concentration_corrected"].attrs["methods"] = "filtered by shape parameter & terminal velocity following Tokay et al 2013"
        # ds["rain_intensity_corrected"].attrs["methods"] = "calculated from dve_fs_bin_concentration_corrected following Tokay et al 2013"
        # ds["rain_intensity"].attrs["methods"] = "rain intensity filtered by wind direction"
        # ds["diameter_ve"].attrs["units"] = "mm"
        # ds["diameter_ve"].attrs["description"] = "  diameter of the hydrometeors"
        # ds["fall_speed_ms"].attrs["units"] = "m/s"
        # ds["fall_speed_ms"].attrs["description"] = "fall speed of the hydrometeors"

        result["dve_bin_concentration_filtered"].attrs["methods"] = "filtered by shape parameter following Tokay et al 2013"
        result["dve_fs_bin_concentration_filtered2"].attrs["methods"] = "filtered by shape parameter & terminal velocity following Tokay et al 2013"
        result["dve_fs_bin_concentration_filtered"].attrs["methods"] = "filtered by terminal velocity only following Tokay et al 2013"
        result["rain_intensity_corrected"].attrs["methods"] = "calculated from dve_fs_bin_concentration_corrected following Tokay et al 2013"
        result["rain_intensity"].attrs["methods"] = "rain intensity filtered by wind direction"
        result["diameter_ve"].attrs["units"] = "mm"
        result["diameter_ve"].attrs["description"] = "  diameter of the hydrometeors"
        result["fall_speed_ms"].attrs["units"] = "m/s"
        result["fall_speed_ms"].attrs["description"] = "fall speed of the hydrometeors"
        result['time'].attrs["long_name"] = "Time UTC"

        result.to_netcdf(new_filename)

    return 


def main(in_dir: str, out_dir: str):

    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(in_dir):
        if filename.endswith(".nc"):
            in_path = os.path.join(in_dir, filename)
            new_filename = os.path.join(out_dir, os.path.splitext(filename)[0] + '_corrected.nc')
            try:
                correct_file(in_path, new_filename)
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    print('processing complete')

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
    args = parser.parse_args()

    main(args.in_dir, args.out_dir)