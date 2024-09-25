#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:35:53 2024

@author: catcoll

variogram
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skgstat import Variogram, DirectionalVariogram
import skgstat as skg



ds=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc")



#%% Long term annual mean over gulf coast region (2018-2022)

gulf=ds.sel(lat=slice(18,42), lon=slice(258,282))

gulf.deltad_weighted
#%% variogram

# delta d values
annual = gulf.deltad_weighted.mean(dim='time')
annual.coords['lon'] = (annual.coords['lon'] + 180) % 360 - 180
# Set spatial dimensions
annual = annual.rio.set_spatial_dims('lon', 'lat')

tropo=annual.values
lon = annual.coords['lon'].values
lat = annual.coords['lat'].values

lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

# Flatten the grids and the tropo data
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()
tropo_flat = tropo.flatten()

data = np.column_stack([lon_flat, lat_flat, tropo_flat])

# Define the coordinates and values
coords = data[:, :2]  # Longitude and latitude
values = data[:, 2]   # Tropo data

v = Variogram(coords, values, model='spherical', bin_func='')

v.fit()

v.plot()


dv = DirectionalVariogram(coords, values, azimuth = 90)
dv.fit()
dv.plot()

dv1 = DirectionalVariogram(coords, values, azimuth = 0)
dv1.fit()
dv1.plot()

#%%














