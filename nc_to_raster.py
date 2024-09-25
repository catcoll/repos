#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:02:00 2024

@author: catcoll

netcdf to raster and averaging
"""


import xarray as xr

#tropomi/nc file
ds=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc")


#slice by region
gulf=ds.sel(lat=slice(25,32), lon=slice(258,282))
#select variable
dw = gulf.deltad_weighted
#%%
"""
3 ways to average

long term average (average over all available time steps)
seasonal
annual

"""
#average over all data, this removes the time dimension, so the out put is a 2d xarray
all_data = dw.mean(dim='time')
all_data.coords['lon'] = (all_data.coords['lon'] + 180) % 360 - 180 #convert lon values to -180 to 180 if you want
all_data = all_data.rio.set_spatial_dims('lon', 'lat') # Set spatial dimensions


#if you want to average by season
seasonal=dw.groupby("time.season").mean()
seasonal= seasonal.sel(season=["DJF", "MAM", "JJA", "SON"]) #reorder seasons


#if you want to average by year
annual=dw.groupby("time.year").mean()


#%%
"""
assign spatial dimensions based on original nc file 

can be necessary if you converted the nc to an array without dimensions, 
this will assign them based on a reference file or original nc data

"""
#assign spatial reference to np array
aligned_xarr = xr.DataArray("input array", dims=annual.dims, coords=annual.coords) 
aligned_xarr = aligned_xarr.rio.set_spatial_dims('lon','lat') # i am not sure why you have to set spatial dims in the two different ways, may not always be necessary


#save as raster
aligned_xarr.rio.to_raster(r"OLS_summer.tif")