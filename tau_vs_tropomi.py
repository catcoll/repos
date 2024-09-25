#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:06:46 2024

@author: catcoll

In this code:
    
    Makes plot of tau bar value from attenutation model vs tropomi delta values
    or makes plots of h2o column data vs tropomi delta values

"""



import xarray as xr
import matplotlib.pyplot as plt
# from rasterio.mask import mask
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression




#load data

y=r'/Users/catcoll/Documents/py/isotopes/atten_tropomi_v3.nc'
dm= xr.open_dataset(y)

x = r'/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc'
ds = xr.open_dataset(x)

#%%

#slice for gulf region
gulf=ds.sel(lat=slice(21,34), lon=slice(260,279))
gulf2=dm.sel(lat=slice(21,34), lon=slice(260,279))

#slice for the same monthly values
a=gulf.isel(time=slice(2,-12))

#reassign month values to model data
b=gulf2
new_time_values = pd.date_range('2018-01-01', '2023-12-01', freq='MS')  # create array of monthly values from 2018-2023
new_time_values
b = b.assign_coords(time=new_time_values)

#model data has 2 measurements for some months, so average by month
deltad=a.deltad_weighted
tau=b.tau_bar
deltad
tau
q = a.h2o_column
#regrid to have the same size arrays

# lat=tau.lat
# lon=tau.lon

# delta_regrid=deltad.interp(lat=lat, lon=lon)
deltad=deltad.values
# tau=tau.values
q=q.values

#flatten for plotting
deltad_flat = deltad.flatten()
tau_flat = tau.flatten()
q_flat = q.flatten()

# Remove NaNs from the data
mask = ~np.isnan(deltad_flat) & ~np.isnan(q_flat) #& ~np.isnan(tau_flat)
deltad_flat = deltad_flat[mask]
tau_flat = tau_flat[mask]
q_flat = q_flat[mask]

# Perform linear regression
model = LinearRegression()
tau_flat_reshaped = tau_flat.reshape(-1, 1)  # Reshape for sklearn
q_flat_reshaped = q_flat.reshape(-1,1)
# model.fit(tau_flat_reshaped, deltad_flat)  # Fit with tau as x and deltad as y
model.fit(q_flat_reshaped, deltad_flat)

# Predict the best fit line
tau_fit = np.linspace(np.min(tau_flat), np.max(tau_flat), 100)
q_fit = np.linspace(np.min(q_flat), np.max(q_flat), 100)
deltad_fit = model.predict(q_fit.reshape(-1, 1))

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(q_flat, deltad_flat, alpha=0.5, label='Data')
plt.plot(q_fit, deltad_fit, color='red', linestyle='--', label='Line of Best Fit')
plt.xlabel('Q')
plt.ylabel('TROPOMI deltad values')
plt.title('Observed vs Modeled Data (Monthly Averages)')
plt.legend()
plt.grid(True)
plt.show()

#%% long term annual mean

#use datasets that have already been slice to the same size

#average over all months to get the long term average
dm_annual=b.tau_bar.mean(dim='time')
dm_annual.shape
ds_annual=a.deltad_weighted.mean(dim='time')
ds_annual.shape
q_annual = a.h2o_column.mean(dim='time')
q_annual.shape
#interpolate to the same grid
lat=dm_annual.lat
lon=dm_annual.lon
delta_a_regrid=ds_annual.interp(lat=lat, lon=lon)

deltad_a=ds_annual.values
tau_a=dm_annual.values
q_a=q_annual.values

# Flatten arrays
deltad_a_flat = deltad_a.flatten()
tau_a_flat = tau_a.flatten()
q_a_flat = q_a.flatten()

# Remove NaNs from the data
mask = ~np.isnan(deltad_a_flat) & ~np.isnan(q_a_flat)#& ~np.isnan(tau_a_flat)
deltad_a_flat = deltad_a_flat[mask]
tau_a_flat = tau_a_flat[mask]
q_a_flat = q_a_flat[mask]

# Perform linear regression
model = LinearRegression()
tau_a_flat_reshaped = tau_a_flat.reshape(-1, 1)  # Reshape for sklearn
q_a_flat_reshaped = q_a_flat.reshape(-1, 1)
model.fit(q_a_flat_reshaped, deltad_a_flat)  # Fit with tau_a as x and deltad_a as y

# Predict the best fit line
tau_fit = np.linspace(np.min(tau_a_flat), np.max(tau_a_flat), 100)
q_fit = np.linspace(np.min(q_a_flat), np.max(q_a_flat), 100)
deltad_fit = model.predict(q_fit.reshape(-1, 1))

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(q_a_flat, deltad_a_flat, alpha=0.5, label='Data')
plt.plot(q_fit, deltad_fit, color='red', linestyle='--', label='Line of Best Fit')
plt.xlabel('Q')
plt.ylabel('TROPOMI deltad values')
plt.title('Long Term Annual Average')
plt.legend()
plt.grid(True)
plt.show()


#%% seasonal mean (all plot on the same graph in this cell)
#use next cell for better plotting


#seasonal
ds_seasonal=a.groupby("time.season").mean()
dm_seasonal=b.groupby("time.season").mean()

ds_seasonal = ds_seasonal.sel(season=["DJF", "MAM", "JJA", "SON"])
dm_seasonal = dm_seasonal.sel(season=["DJF", "MAM", "JJA", "SON"])

deltad_s=ds_seasonal.deltad_weighted
deltad_s.shape

q_s = ds_seasonal.h2o_column
q_s.shape

tau_s=dm_seasonal.tau_bar
tau_s


lat=tau_s.lat
lon=tau_s.lon
delta_s_regrid=deltad_s.interp(lat=lat, lon=lon)

deltad_s=deltad_s.values
tau_s=tau_s.values
delta_s_regrid
deltad_s.shape
tau_s.shape
q_s = q_s.values

deltad_s_flat=deltad_s.flatten()
tau_s_flat=tau_s.flatten()
q_s_flat = q_s.flatten()


# Remove NaNs from the data
mask = ~np.isnan(deltad_s_flat) & ~np.isnan(q_s_flat)#~np.isnan(tau_s_flat)
deltad_flat = deltad_s_flat[mask]
q_flat = q_s_flat[mask]
tau_flat = tau_s_flat[mask]

# Perform linear regression
model = LinearRegression()
deltad_flat_reshaped = deltad_flat.reshape(-1, 1)  # Reshape for sklearn
model.fit(deltad_flat_reshaped, q_flat)

# Predict the best fit line
deltad_fit = np.linspace(np.min(deltad_flat), np.max(deltad_flat), 100)
q_fit = model.predict(deltad_fit.reshape(-1, 1))

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(deltad_flat, q_flat, alpha=0.5, label='Data')
plt.plot(deltad_fit, q_fit, color='red', linestyle='--', label='Line of Best Fit')
plt.xlabel('TROPOMI deltad values')
plt.ylabel('Q')
plt.title('Long Term Annual Average')
plt.legend()
plt.grid(True)
plt.show()


#%% iterate over each season


#do not need if previous cell was run, run these if not
# # Compute seasonal averages
# ds_seasonal = gulf.groupby("time.season").mean()
# dm_seasonal = gulf2.groupby("time.season").mean()

# # Select the desired seasons
# ds_seasonal = ds_seasonal.sel(season=["DJF", "MAM", "JJA", "SON"])
# dm_seasonal = dm_seasonal.sel(season=["DJF", "MAM", "JJA", "SON"])

deltad_s = ds_seasonal.deltad_weighted
# tau_s = dm_seasonal.tau_bar
q_s = ds_seasonal.h2o_column
# Regrid data to have the same size arrays
lat = tau_s.lat
lon = tau_s.lon

# Initialize the model
model = LinearRegression()

# Create a figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# List of seasons for easier iteration
seasons = ["DJF", "MAM", "JJA", "SON"]

# Iterate over each season and plot
for ax, season in zip(axes.flat, seasons):
    # Extract data for the current season
    deltad_season = deltad_s.sel(season=season).values
    q_season = q_s.sel(season=season).values
    
    # Flatten the arrays for plotting
    deltad_s_flat = deltad_season.flatten()
    q_s_flat = q_season.flatten()

    # Remove NaNs from the data
    mask = ~np.isnan(deltad_s_flat) & ~np.isnan(q_s_flat)
    q_flat = q_s_flat[mask]  # Use tau for x-axis
    deltad_flat = deltad_s_flat[mask]  # Use deltad for y-axis

    # Perform linear regression
    q_flat_reshaped = q_flat.reshape(-1, 1)  # Reshape for sklearn
    model.fit(q_flat_reshaped, deltad_flat)

    # Predict the best fit line
    q_fit = np.linspace(np.min(q_flat), np.max(q_flat), 100)
    deltad_fit = model.predict(q_fit.reshape(-1, 1))

    # Calculate R^2
    r2 = model.score(q_flat_reshaped, deltad_flat)

    # Plot on the current subplot
    ax.scatter(q_flat, deltad_flat, alpha=0.5, label='Data')
    ax.plot(q_fit, deltad_fit, color='red', linestyle='--', label='Line of Best Fit')
    ax.set_xlabel('Q')
    ax.set_ylabel('TROPOMI deltad values')
    ax.set_title(f'Season: {season}\n$R^2 = {r2:.2f}$')
    ax.legend()
    ax.grid(True)

plt.show()











