#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:12:26 2024

@author: catcoll

plot singular orbits and time for that orbit
"""


import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import gcsfs

xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)

%xmode minimal
%matplotlib inline
%config InlineBackend.figure_format='retina'



#%% extract data from one orbit
# select file, gulf coast is usually the 4th to last orbit on the folder
x="/Volumes/waterisotopes/HDO_trim_CLEAR/2021/11/11/s5p_l2_hdo_0003_21145_CLEAR_trim.nc"

#open by group to get data and coordinates
obj=xr.open_dataset(x, group="target_product")
coords=xr.open_dataset(x, group="instrument")
coords.time
time=coords["time"].values
time[0]
#extract data
deltad=obj['deltad'].values
lon=coords['longitude_center'].values
lat=coords['latitude_center'].values

lon2d, lat2d = np.meshgrid(lon, lat)
deltad.shape
houston=(265,29)
type(lat)
lon2d.shape
lat.shape
deltad = deltad.reshape(len(lat), len(lon))



#%% plot one orbit

#need data to be a 2D array to use pcolormesh
#types of plots that work as is: tricolor

import cartopy.crs as ccrs

fig, ax=plt.subplots(
    figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
ax.coastlines()
ax.set_extent([260,279,27,34])
img=ax.scatter(lon,lat,c=deltad,cmap="Spectral", transform=ccrs.PlateCarree())
ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
   draw_labels=True, alpha=.04, linestyle='--')
plt.colorbar(img,ax=ax)


ax.scatter(houston[0],houston[1],color='red', marker='o')

#%% plot time

time
hr_min=times[:,4:5]
hr_min
hr_min.shape
#convert time to 1 value



lon.shape
lat.shape
hr_min.shape

fig, ax=plt.subplots(
    figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
ax.coastlines()
ax.set_extent([260,279,27,34])
# Plot using scatter with color mapped by time
sc = ax.scatter(lon, lat, c=hr_min, cmap="plasma", transform=ccrs.PlateCarree())
plt.colorbar(sc, ax=ax, label='Time of Observation')



