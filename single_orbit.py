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
import cartopy.crs as ccrs

xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)

%xmode minimal
%matplotlib inline
%config InlineBackend.figure_format='retina'



#%% extract data from one orbit

x="/Volumes/waterisotopes/HDO_trim_CLEAR/2022/03/20/s5p_l2_hdo_0003_22975_CLEAR_trim.nc"
date=x[38:48]
#open by group to get data and coordinates
obj=xr.open_dataset(x, group="target_product")
coords=xr.open_dataset(x, group="instrument")

#extract data and coordinates
time=coords["time"].values
deltad=obj['deltad'].values
lon=coords['longitude_center'].values
lat=coords['latitude_center'].values

#housotn coords, need to update
houston=(265,29)

#%%plot one orbit with scatter
#need data to be a 2D array to use pcolormesh
#types of plots that work as is: tricolor



fig, ax=plt.subplots(
    figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
ax.coastlines()
ax.set_extent([263,267,28,31.5])
img=ax.scatter(lon,lat,c=deltad,cmap="plasma", transform=ccrs.PlateCarree(), vmin=-0.35, vmax=-0.05)
ax.set_title("delta values for "+date, y=1.2)
ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
   draw_labels=True, alpha=.04, linestyle='--')
plt.colorbar(img,ax=ax)


ax.scatter(houston[0],houston[1],color='yellow', marker='o', edgecolors="black")

#%% check for time when measurments are taken


hr_min=time[:,3:6]
#hr, min, sec
time[-1]
time[0]
hr_min.shape
type(hr_min)

# Example numpy arrays with hour, minute, second data
hours = hr_min[:,0]
minutes = hr_min[:,1]
seconds = hr_min[:,2]

obs_time_s=minutes*60+seconds
obs_time_m=minutes+seconds/60

#%% plot time by seconds fter UTC hour (not as useful as minutes)



# fig, ax=plt.subplots(
#     figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
# ax.coastlines()
# ax.set_extent([260,279,27,34])
# # Plot using scatter with color mapped by time
# sc = ax.scatter(lon, lat, c=obs_time_s, cmap="plasma", transform=ccrs.PlateCarree(), vmin=2560, vmax=2700)
# ax.set_title("Seconds past 1pm CT (19 UTC)", y=1.2)
# ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
#    draw_labels=True, alpha=.04, linestyle='--')
# plt.colorbar(sc, ax=ax, label='Time of Observation')


# np.where(obs_time_m==23)

#%% plot minutes past UTC hour

fig, ax=plt.subplots(
    figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
ax.coastlines()
ax.set_extent([260,279,27,34])
# Plot using scatter with color mapped by time
sc = ax.scatter(lon, lat, c=obs_time_m, cmap="plasma", transform=ccrs.PlateCarree(), vmin=36, vmax=38)
ax.set_title("Minutes past 1pm CT (19 UTC)", y=1.2)
ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
   draw_labels=True, alpha=.04, linestyle='--')
plt.colorbar(sc, ax=ax, label='Time of Observation')



#%% extracting data for reformatting to work with pcolormesh plotting

x="/Volumes/waterisotopes/HDO_trim_CLEAR/2022/03/20/s5p_l2_hdo_0003_22975_CLEAR_trim.nc"

#open groups
obj=xr.open_dataset(x, group="target_product")
coords=xr.open_dataset(x, group="instrument")
obj
coords
#assign variables as 1D arrays
deltad=obj['deltad']
lon_values=coords['longitude_center'].data
lat_values=coords['latitude_center'].data
type(lon_values)

print(f'lon_values.shape: {lon_values.shape}')
print(f'lat_values.shape: {lat_values.shape}')
print(f'deltad.shape: {deltad.shape}')
print(f'lon_values.dtype: {lon_values.dtype}')
print(f'lat_values.dtype: {lat_values.dtype}')
print(f'deltad.dtype: {deltad.dtype}')



#%% attempting to add dimension to deltad variable for plotting, not successful

#create data array
a=xr.DataArray(deltad)
a
#expand dimensions
a=a.expand_dims(dim=dict(
    lon=(lon_values),  
    lat=(lat_values)))

# a=a.expand_dims(dim=(["lon","lat"]))
a.variables
#add coords
# a=xr.DataArray(deltad, coords={"lon": lon_values, "lat": lat_values})

# #trying again in a different way
# a = xr.DataArray(deltad, coords={"lon": lon_values, "lat": lat_values, "nobs": nobs})

#%% creating xarray dataset with coords and data variable deltad

#dataset
ds = xr.Dataset(
    data_vars=dict(
        a=(a),
    ),
    coords=dict(
        lon=(lon_values),  
        lat=(lat_values),  
    ),
    attrs=dict(description="Delta expression of HDO to H2O ratio"),
)
ds
deltad=ds.a
lon=ds.lon.data
lon
lat=ds.lat.data
lon
lat1=27
lat2=34
lon1=80
lon2=98

lat1_idx = np.argmin(np.abs(ds.lat.data - lat1))
lat2_idx = np.argmin(np.abs(ds.lat.data - lat2))
# lon1_idx = np.argmin(np.abs(ds.lat.data - lon1))
# lon1_idx
# lon2_idx = np.argmin(np.abs(ds.lat.data - lon2))
# lon2_idx
gulf=ds.isel(lat=slice(lat1_idx,lat2_idx))
gulf
# f, ax = plt.subplots(1,2, sharex=True, sharey=True)
# ax[0].tripcolor(lon,lat,deltad)
# ax[1].tricontourf(lon,lat,deltad, 20) 
# ax[1].plot(lon,lat, 'ko ')
# ax[0].plot(lon,lat, 'ko ')
# ax.set_extent([260,279,27,34])

plt.savefig('test.png')

plt.pcolormesh(gulf.lon.data, gulf.lat.data, gulf.a)


fig, ax=plt.subplots(
    figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
ax.coastlines()
ax.set_extent([263,267,28,31.5])
img=ax.scatter(lon,lat,c=deltad,cmap="plasma", transform=ccrs.PlateCarree(), vmin=-0.35, vmax=-0.05)
ax.set_title("delta values for "+date, y=1.2)
ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
   draw_labels=True, alpha=.04, linestyle='--')
plt.colorbar(img,ax=ax)


ax.scatter(houston[0],houston[1],color='yellow', marker='o', edgecolors="black")






