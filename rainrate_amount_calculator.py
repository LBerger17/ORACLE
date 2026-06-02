#!/bin/python

import os
import numpy as np
import xarray as xr
import scipy.stats as stats
import argparse

parser = argparse.ArgumentParser(description='Provide precipitation file and pdf details')
parser.add_argument('--in_file', type=str,
                    help='a string specifying the precipitation file')
parser.add_argument('--out_file', type=str,
                    help='a string specifying the amount distribution file')
parser.add_argument('--smallest_bin_center', type=float, default=0.03,
                    help='a float specifying the smallest bin center of the pdf')
parser.add_argument('--growth_rate', type=float, default=0.07,
                    help='a float specifying the growth rate of the pdf')
parser.add_argument('--precip_name', type=str, default='PRECT',
                    help='a string specifying the variable name for precipitation')
parser.add_argument('--start_time', type=str, default=None,
                    help='specify the desired start time within the precip file')
parser.add_argument('--end_time',   type=str, default=None,
                    help='specify the desired end time within the precip file')
parser.add_argument('--month_mask', type=int, nargs='+', default=None, 
                    help='specify which months to include (as list)')
args = parser.parse_args()

in_file       = args.in_file
out_file      = args.out_file
c1            = args.smallest_bin_center
growth_rate   = args.growth_rate
precip_name   = args.precip_name
start_time    = args.start_time
end_time      = args.end_time
if args.month_mask is None:
    month_mask = np.arange(1,13)
    print('Using all times')
else:
    month_mask = args.month_mask
    print('Using only months with indexes in', month_mask)

fac           = 1. + growth_rate
x1            = np.sqrt(c1**2/fac)                # center to edge factor
cntrs         = np.append([0.], c1*fac**np.arange(0,151))
edges         = np.append([0.], x1*fac**np.arange(0,152))
iwidth        = 1. / growth_rate

def convert_PRECT_units(PRECT, desired_units):
    if(PRECT.attrs['units'].lower()=='mm/day' and desired_units.lower()=='kg/m2/s'):
        PRECT = PRECT / 86400.
    elif(PRECT.attrs['units'].lower()=='kg/m2/s' and desired_units.lower()=='mm/day'):
        PRECT = PRECT * 86400.
    elif(PRECT.attrs['units'].lower()=='m/s' and desired_units.lower()=='kg/m2/s'):
        PRECT = PRECT * 1e3
    elif(PRECT.attrs['units'].lower()=='m/s' and desired_units.lower()=='mm/day'):
        PRECT = PRECT * 86400e3
    PRECT.attrs.update({'units': desired_units})
    return PRECT

def use_month(month, month_mask=np.arange(1,13)):
    return np.isin(month, month_mask)

with xr.open_dataset(in_file) as data:
    try:
        data = data.sel(time=slice(start_time, end_time))
    except KeyError:
        print("Time slicing only valid if 'time' is a dimension within the dataset")
    data   = data.load()
    precip = convert_PRECT_units(data[precip_name], desired_units='mm/day').compute()
    ds     = xr.Dataset()

    if ('lon' in data.dims) and ('lat' in data.dims):
        amounts = np.zeros([len(cntrs), len(data['lon']), len(data['lat'])])
        for m, lon in enumerate(data['lon']):
            for n, lat in enumerate(data['lat']):
                prec_ts        = precip.sel(lon=lon, lat=lat, 
                                            time=use_month(precip['time.month'], month_mask))
                stat_metric    = stats.binned_statistic(prec_ts, prec_ts, statistic='sum', bins=edges)
                amounts[:,m,n] = iwidth * stat_metric.statistic / prec_ts.size
        ds['amount'] = xr.DataArray(amounts, 
                                    coords={'centers':cntrs, 'lon':data['lon'], 'lat':data['lat']},
                                    dims=('centers', 'lon', 'lat'))

    if ('ncol' in data.dims):
        amounts = np.zeros([len(cntrs), len(data['ncol'])])
        for n, ncol in enumerate(data['ncol']):
            prec_ts      = precip.sel(ncol=ncol,
                                      time=use_month(precip['time.month'], month_mask))
            stat_metric  = stats.binned_statistic(prec_ts, prec_ts, statistic='sum', bins=edges)
            amounts[:,n] = iwidth * stat_metric.statistic / prec_ts.size
        ds['amount'] = xr.DataArray(amounts, 
                                    coords={'centers':cntrs, 'ncol':data['ncol']},
                                    dims=('centers', 'ncol'))

    ds.coords['edges'] = ('edges', edges)
    ds.attrs.update({'Smallest bin center': c1,
                     'growth factor percentage': fac,
                     'center to edge factor': x1,
                     'inverse_width': iwidth,
                     'author': 'Bryce Harrop'})
    ds.to_netcdf(out_file, mode='w')
    
