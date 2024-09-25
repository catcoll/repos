#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:26:06 2024

@author: catcoll

plot and save all tracer days around houston
plot time of measurement

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


# read in files as indexable list
with open("/Users/catcoll/Documents/py/isotopes/tracer_filescopy.txt") as f:
    files = [line.rstrip('\n') for line in f]
# with open("/Users/catcoll/Documents/py/isotopes/tracer_files_juntosept_22.txt") as f:
#     files2 = [line.rstrip('\n') for line in f]
files[0]


#%% create function that opens netcdf4 file, plots houston region, and saves file to folder 


def tracer(files):
    
    for i, x in enumerate(files):
        try:
            date=x[38:48]
            date=date.replace('/', "-")
            obj=xr.open_dataset(x, group="target_product")
            coords=xr.open_dataset(x, group="instrument")
            coords
            time=coords["time"].values
            #extract data
            deltad=obj['deltad'].values
            lon=coords['longitude_center'].values
            lat=coords['latitude_center'].values
            houston=(265,29)
            
            
            fig, ax=plt.subplots(
                figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
            ax.coastlines()
            ax.set_extent([263,267,28,31.5])
            img=ax.scatter(lon,lat,c=deltad,cmap="plasma", transform=ccrs.PlateCarree(), vmin=-0.3, vmax=-0.0)
            ax.set_title("delta values for "+date, y=1.2)
            ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
               draw_labels=True, alpha=.04, linestyle='--')
            plt.colorbar(img,ax=ax)
            ax.scatter(houston[0],houston[1],color='yellow', marker='o', edgecolors="black")
            
            #save to folder
            fig.savefig("/Users/catcoll/Documents/py/isotopes/h_tracer_plots"+"/"+"delta_values_for_"+date+"_"+str(i)+'.png')
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return fig

#%% run fxn

plots=tracer(files)

#%% plot time
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

    
    
    
    
    