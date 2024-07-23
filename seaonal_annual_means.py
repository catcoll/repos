# -*- coding: utf-8 -*-
"""
Cat Collins
06/20/24

Summary:
Import NETCDF files with xarray, extract delta d data, constrain to guld region,
plot on world map using cartopy
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
ds=xr.open_dataset("/Users/catcoll/Documents/py/isotopes/TROPOMI_HDO_CLEAR_1x1.nc")
# ds.to_netcdf("/Users/catcoll/Documents/test_full.nc", "w","NETCDF4")
delta=ds.deltad
# ds.variables.keys()


# delta.to_netcdf("/Users/catcoll/Documents/test1.nc", "w","NETCDF4")
#dd is: delta expression of HDO to H2O ratio
dd=ds.deltad

#Water mass weighted delta expression of HDO to H2O ratio
dw=ds.deltad_weighted

dw
dw.shape
#%% plot first time step

lat=dd.lat.data
#np.where(lat==27) #117
#lat=lat[117:125]
lon=dd.lon.data
#lon=lon[260:279]

temp=dd.data
plt.figure()
plt.pcolormesh(lon,lat,temp[0,:,:])
temp.mean(axis=1)

dd.isel(time=1).plot(x="lon")



#%% average weighted delta fraction over years and seasons

seasonal_mean=ds.groupby("time.season").mean()
seasonal_mean

seasonal_mean = seasonal_mean.sel(season=["DJF", "MAM", "JJA", "SON"])
seasonal_mean

#seasonal_mean.deltad_weighted.plot(col="season", col_wrap=2)

annual_mean=ds.groupby("time.year").mean()
annual_mean

#annual_mean.deltad_weighted.plot(col="year", col_wrap=3)

#%% index for lat lon of the gulf coast region

# "nearest indexing at multiple points"
gulf=ds.sel(lat=slice(21,34), lon=slice(260,279))
gulf
gulf_annual=gulf.groupby("time.year").mean()
#gulf_annual.deltad_weighted.plot(col="year", col_wrap=3)
gulf_a_dw=gulf_annual.deltad_weighted


gulf_seasonal=gulf.groupby("time.season").mean()
gulf_seasonal = gulf_seasonal.sel(season=["DJF", "MAM", "JJA", "SON"])
gulf_seasonal
#gulf_seasonal.deltad_weighted.plot(col="season", col_wrap=2)

gulf_s_dw=gulf_seasonal.deltad_weighted
gulf_s_dw






#%% Remove 2017

gulf_a_dw
gulf_a_dw=gulf_a_dw.sel(year=slice(2018,2022))

seasons=gulf_s_dw.season.values
seasons=seasons.tolist()
type(seasons)
seasons[0]


#%% regular plotting

# fig, ax=plt.subplots(
#     figsize=(10,5), subplot_kw={"projection":ccrs.PlateCarree()})
# ax.coastlines()
# #ax.set_extent([260,279,27,34])
# # Plot using scatter with color mapped by time
# fig, ax = gulf_s_dw.plot.pcolormesh(col='season', cmap="plasma", transform=ccrs.PlateCarree(), vmin=36, vmax=38)
# ax.set_title("Minutes past 1pm CT (19 UTC)", y=1.2)
# ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', 
#    draw_labels=True, alpha=.04, linestyle='--')
# plt.colorbar(sc, ax=ax, label='Time of Observation')

ps = gulf_s_dw.plot.pcolormesh(
    col="season", col_wrap=3,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    norm = mpl.colors.Normalize(vmin=-.3, vmax=-0.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,21,34])
ax.set_title("Annual Average of Mass Weighted Delta d Ratio (1deg)", y=3.3)

pa = gulf_a_dw.plot.pcolormesh(
    col="year", col_wrap=3,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    norm = mpl.colors.Normalize(vmin=-.3, vmax=-0.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in pa.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,279,27,34])
ax.set_title("Annual Average of Mass Weighted Delta d Ratio (1deg)", y=3.3)


#%% Same but with p05 data

ds05=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p05.nc")
ds05
gulf05=ds05.sel(lat=slice(27,34), lon=slice(260,279))
gulf05

dw05=gulf05.deltad_weighted
dw05
seasonal_mean05=dw05.groupby("time.season").mean()
seasonal_mean05 = seasonal_mean05.sel(season=["DJF", "MAM", "JJA", "SON"])
#seasonal_mean05.plot(col="season", col_wrap=2)

annual_mean05=dw05.groupby("time.year").mean()
#annual_mean05.plot(col="year", col_wrap=3)
#annual_mean05=annual_mean05.sel(year=slice(2018,2022))
annual_mean05

#%% remove 2017

annual_mean05=annual_mean05.sel(year=slice(2018,2022))
annual_mean05

#%% cartopy with p05

p05_s = seasonal_mean05.plot.pcolormesh(
    col="season", col_wrap=2,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    norm = mpl.colors.Normalize(vmin=-.3, vmax=-0.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in p05_s.axs.flat:
    ax.coastlines()
    ax.gridlines()
#ax.set_title("Seasonal Average of Mass Weighted Delta d Ratio (5km)", y=1.2)

p05_a = annual_mean05.plot.pcolormesh(
    col="year", col_wrap=3,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    norm = mpl.colors.Normalize(vmin=-.3, vmax=-0.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
plt.title("Annual Average of Mass Weighted Delta d Ratio (5km)", y=2.3)
for ax in p05_a.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,279,27,34])
ax.set_title("Seasonal Average of Mass Weighted Delta d Ratio (5km)", y=1.2)



#%% # get obvs per year (not finished)

years=[np.datetime_as_string(date,unit='Y').astype(int) for date in t]

n=gulf.nobs.data
nobvs_y=np.stack((years,n))
nobvs_y[1,1]


year_2017=0
year_2018=0
year_2019=0
year_2020=0
year_2021=0
year_2022=0

# NOT FINISHED

for i in range(len(nobvs_y[0])):
   # year = nobvs_m[0,10]
    value = nobvs_m[1,i]
    
    if nobvs_y[0,i]== 1 or nobvs_m[0,i]== 2 or nobvs_m[0,i]== 12:
        winter += value
        
    if nobvs_m[0,i]== 3 or nobvs_m[0,i]== 4 or nobvs_m[0,i]== 5:
        spring += value
        
    if nobvs_m[0,i]== 6 or nobvs_m[0,i]== 7 or nobvs_m[0,i]== 8:
       summer += value
        
    if nobvs_m[0,i]== 9 or nobvs_m[0,i]== 10 or nobvs_m[0,i]== 11:
       fall += value
print(f"Winter: {winter}")
print(f"Spring: {spring}")
print(f"Summer: {summer}")
print(f"Fall: {fall}")

winter
#%% # get obs for each season
from datetime import date
t=gulf.time.data
type(t)
t.datetime64

months = np.array([np.datetime64(date, 'M').astype('int').item() % 12 + 1 for date in t])

months
n=gulf.nobs.data
nobvs_m=np.stack((months,n))
nobvs_m[1][0]


winter=0
summer=0
spring=0
fall=0
for i in range(len(nobvs_m[0])):
 #   month = nobvs_m[0,i]
    value = nobvs_m[1,i]
    
    if nobvs_m[0,i]== 1 or nobvs_m[0,i]== 2 or nobvs_m[0,i]== 12:
        winter += value
        
    if nobvs_m[0,i]== 3 or nobvs_m[0,i]== 4 or nobvs_m[0,i]== 5:
        spring += value
        
    if nobvs_m[0,i]== 6 or nobvs_m[0,i]== 7 or nobvs_m[0,i]== 8:
       summer += value
        
    if nobvs_m[0,i]== 9 or nobvs_m[0,i]== 10 or nobvs_m[0,i]== 11:
       fall += value
print(f"Winter: {winter}")
print(f"Spring: {spring}")
print(f"Summer: {summer}")
print(f"Fall: {fall}")

winter
#%% fancy plotting with advanced labelling stuff

import cartopy.crs as ccrs
import matplotlib as mpl




fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.Orthographic(-80, 45)})
axs = axs.flatten()

seasons = ["DJF", "MAM", "JJA", "SON"]
observations = [winter, spring, summer, fall]

for i, (season, num_observations) in enumerate(zip(seasons, observations)):
    ax = axs[i]
    gulf_s_dw.sel(season=season).plot.pcolormesh(
        transform=ccrs.PlateCarree(),
        cmap='Spectral',
        norm=mpl.colors.Normalize(vmin=-0.3, vmax=-0.05),
        ax=ax
    )
    ax.coastlines()
    ax.gridlines()
    ax.set_title(f"{season}\nN Obs: {num_observations}")
#%% other plotting wit number of observations included


import cartopy.crs as ccrs
import matplotlib as mpl


#with nobs variable
nobs=gulf.nobs.data
nobs
pa = dw.plot.pcolormesh(
    col="time", col_wrap=4,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    norm = mpl.colors.Normalize(vmin=-.3, vmax=-0.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax, n_obs in zip(pa.axes.flat, nobs):
    ax.coastlines()
    ax.gridlines()
    ax.set_extent([260,279,27,34])
    ax.text(0.95, 0.95, f"N Obs: {n_obs}", transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
 
    
    
# with count variable
# turns out werid because each input is an array of values

count=ds['count']
count=count.data
count
#count.values
pa = dw.plot.pcolormesh(
    col="time", col_wrap=4,
    transform=ccrs.PlateCarree(), cmap='Spectral',
    norm = mpl.colors.Normalize(vmin=-.3, vmax=-0.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax, ct in zip(pa.axes.flat, count):
    ax.coastlines()
    ax.gridlines()
    ax.set_extent([260,279,27,34])
    ax.text(0.95, 0.95, f"N Obs: {ct}", transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))  


#%%

ds=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc")
ds.to_netcdf("/Users/catcoll/Documents/test_full.nc", "w","NETCDF4")
delta=ds.deltad
ds.variables.keys()
delta=delta.transpose()
delta=delta.transpose("lon","lat","time")
delta.to_netcdf("/Users/catcoll/Documents/test1.nc", "w","NETCDF4")














