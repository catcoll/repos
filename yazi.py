#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:17:57 2024

@author: catcoll

Look at the YAZI sattellite
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import gcsfs
import cartopy.crs as ccrs
import matplotlib as mpl

xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)

%xmode minimal
%matplotlib inline
%config InlineBackend.figure_format='retina'

#%% Upload data
x= "/Volumes/waterisotopes/10.35097-495/data/dataset/2014/IASIAB_MUSICA_030201_L3pp_H2Oiso_v2_20141001_evening_global.nc"
ds=xr.open_dataset(x)
dd=ds.musica_deltad
#make lat and lon the same length
#dd=dd.sel(lon=slice(-179,1))
lat=dd.lat.data
lon=dd.lon.data
len(lat)
len(lon)
# Create a meshgrid of latitude and longitude
lon_mesh, lat_mesh = np.meshgrid(lon, lat)
lat_mesh.shape
dd1=dd[0,:,:]
dd1
dd.shape
#%% Plotting the whole data set
plt.figure(figsize=(10, 6))
plt.contourf(lon_mesh, lat_mesh, dd1.values, cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Example Plot with Latitude and Longitude')
plt.grid(True)
plt.show()

lat_mesh[0,0]
lon_mesh


#%% trying to select by lat lon
lon_mesh, lat_mesh = np.meshgrid(lon, lat)
lon_mesh
for i in range(len(lon_mesh)):
    for j in range(len(lat_mesh)):
        
        x = lon_mesh[i, j]
        y = lat_mesh[i, j]
            
            # Example condition: x between -100 and -50, y between 30 and 60
        if 0 <= x <= 98 and 28 <= y <= 100:
            print("yes")
    else:
        print("no")



#%%plot 1 time step
# temp=dd.data
# plt.figure()
# plt.pcolormesh(lon,lat,temp[0,:,:])
# temp.mean(axis=1)

# dd.isel(time=1).plot(x="lon")




#%% plot houston w coastlines

# ax.scatter(265.3698,29.7604,color='yellow', marker='o', edgecolors="black")
ps = dd.plot.pcolormesh(
    col="altitude_levels", col_wrap=3,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    #norm = mpl.colors.Normalize(vmin=-.3, vmax=-0.05),
    subplot_kws={'projection': ccrs.PlateCarree()}
    )
for ax in ps.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([263,267,28,31.5])


#%% try to pull out files names of data around houston


import os

def get_iasi(start_dir):
    
    file_list=[]
    
    for files in os.walk(start_dir):

        for filename in os.listdir(start_dir):
            file_path = os.path.join(start_dir, filename)
            file_list.append(file_path)

    return file_list


#%%

iasi_2014=get_iasi("/Volumes/waterisotopes/10.35097-495/data/dataset/2014")
iasi_2014
a=iasi_2014[0]
date=x[-26:-3]
date
#%%


def iasi(files):
    
    for x in files:
        date=x[-26:-3]
        ds=xr.open_dataset(x)
        dd=ds.musica_deltad
        #make lat and lon the same shape
        #dd=dd.sel(lon=slice(-179,1))
        lat=dd.lat.data
        lon=dd.lon.data

        # Create a meshgrid of latitude and longitude
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        lat_mesh.shape

        
        fig = plt.figure(figsize=(17, 10))
        grid = dd.plot.pcolormesh(
            col="altitude_levels", col_wrap=3,
            transform=ccrs.PlateCarree(), cmap='Spectral',
            subplot_kws={'projection': ccrs.PlateCarree()}
        )
        
        # Customize each subplot
        for i, ax in enumerate(grid.axs.flat):
            ax.coastlines()
            ax.gridlines()
            ax.set_extent([263, 267, 28, 31.5])
        ax.set_title(date)

                
        #save to folder
        filename = f"/Users/catcoll/Documents/py/isotopes/iasi_plots/delta_values_for_{date}.png"
        plt.savefig(filename)
        plt.close()
    
    return fig

#%%

iasi(iasi_2014)

fig = plt.figure(figsize=(15, 10))
grid = dd.plot.pcolormesh(
    col="altitude_levels", col_wrap=3,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    subplot_kws={'projection': ccrs.PlateCarree()}
)

# Customize each subplot
for i, ax in enumerate(grid.axs.flat):
    ax.coastlines()
    ax.gridlines()
    ax.set_extent([263, 267, 28, 31.5])
ax.set_title("delta values for "+date, y=1.2)


fig



