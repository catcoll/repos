#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:01:50 2024

@author: catcoll

In this code:
    input variables averages over time
    gaussian smoothing
    code to save data to use in R and to open data created in R
    OLS regression with scaling
    converting h2o column units
    lots of plotting
    saving residual as a raster
    

"""


import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import Avogadro
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




ds=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc")



#%% Long term annual mean over gulf coast region (2018-2022)

gulf=ds.sel(lat=slice(25,32), lon=slice(258,282))

dw=gulf.deltad_weighted
#%% delta d values
annual = gulf.deltad_weighted.mean(dim='time')
annual.coords['lon'] = (annual.coords['lon'] + 180) % 360 - 180
# Set spatial dimensions
annual = annual.rio.set_spatial_dims('lon', 'lat')

tropo=annual.values

#gaussian smoothing
delta_smooth= gaussian_filter(tropo, sigma = 3, mode='nearest')

#by season
annual = gulf.deltad_weighted.groupby("time.season").mean()
annual.coords['lon'] = (annual.coords['lon'] + 180) % 360 - 180
# # Set spatial dimensions
annual = annual.rio.set_spatial_dims('lon', 'lat')
summer = annual.sel(season='JJA')
summer_data = annual.sel(season='JJA').values
# summer_data=np.flip(summer_data, axis=1)
# summer_data=np.flip(summer_data, axis=0)  #flip for plotting
# summer_data=np.flip(summer_data, axis=1)
winter = annual.sel(season='DJF')
winter_data = annual.sel(season='DJF').values
# winter_data=np.flip(winter_data, axis=1)
# winter_data=np.flip(winter_data, axis=0)  #flip for plotting
# winter_data=np.flip(winter_data, axis=1)

summer_smooth= gaussian_filter(summer_data, sigma = 3, mode='nearest')
winter_smooth= gaussian_filter(winter_data, sigma = 3, mode='nearest')

#%% do same for h2o column data
h2o = gulf.h2o_column.mean(dim='time')
h2o.coords['lon'] = (h2o.coords['lon'] + 180) % 360 - 180
h2o = h2o.rio.set_spatial_dims('lon', 'lat')

h2o=h2o.values
#convert to mass/m2

h2o = h2o* (1/Avogadro) * (0.018/1) * ((100)^2)

#gaussian smoothing
h2o_smooth= gaussian_filter(h2o, sigma = 3, mode='nearest')

#plot both h2o and delta smooth
fig=plt.figure()
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax.imshow(delta_smooth)
ax2.imshow(h2o_smooth)
plt.show()

#by season

h2o = gulf.h2o_column.groupby("time.season").mean()
h2o.coords['lon'] = (h2o.coords['lon'] + 180) % 360 - 180
h2o = h2o.rio.set_spatial_dims('lon', 'lat')
summer_h2o = h2o.sel(season='JJA').values
summer_h2o = summer_h2o* (1/Avogadro) * (0.018/1) * ((100)^2)
# summer_h2o=np.flip(summer_h2o, axis=1)
# summer_h2o=np.flip(summer_h2o, axis=0)  #flip for plotting
# summer_h2o=np.flip(summer_h2o, axis=1)
winter_h2o = h2o.sel(season='DJF').values
winter_h2o = winter_h2o* (1/Avogadro) * (0.018/1) * ((100)^2)
# winter_h2o=np.flip(winter_h2o, axis=1)
# winter_h2o=np.flip(winter_h2o, axis=0)  #flip for plotting
# winter_h2o=np.flip(winter_h2o, axis=1)

summer_h2o_smooth= gaussian_filter(summer_h2o, sigma = 3, mode='nearest')
winter_h2o_smooth= gaussian_filter(winter_h2o, sigma = 3, mode='nearest')


#%% save smoothed data
# pd.DataFrame(h2o_smooth).to_csv('h2o_smooth.csv', index=False, header=False)
# pd.DataFrame(delta_smooth).to_csv('delta_smooth.csv', index=False, header=False)

# dimensions = np.array([h2o_smooth.shape[0], h2o_smooth.shape[1]])
# pd.DataFrame(dimensions).to_csv('dimensions.csv', index=False, header=False)
#%% OLS regression


X_osl = h2o
y_osl = tropo
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_osl, y_osl, test_size=0.2, random_state=42)

#scale
min_max_scaler = MinMaxScaler().fit(X_test)
X_norm = min_max_scaler.transform(X_osl)


# Initialize and fit the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Predict using the test set
pred = reg.predict(X_osl)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test)
plt.ylabel('deltad')
plt.xlabel('h2o column')
plt.title('Linear Regression (OSL), test data')
plt.legend()
plt.show()

# Plot original data and predictions
plt.figure(figsize=(14, 7))

# Plot original delta data
plt.subplot(1, 2, 1)
plt.imshow(y_osl, cmap='viridis',  vmin = -0.12, vmax = -0.21, interpolation='nearest')
plt.colorbar()
plt.title('Original Delta (y_osl)')

# Plot predicted delta data
plt.subplot(1, 2, 2)
plt.imshow(pred, cmap='viridis',  vmin = -0.12, vmax = -0.21, interpolation='nearest')
plt.colorbar()
plt.title('Predicted Delta (OSL)')

plt.tight_layout()
plt.show()


#%% calculate residual

#from osl regression: 
residual=tropo-pred

#%% load rma predictions from R 
# rma_pred_delta = "/Users/catcoll/Documents/py/isotopes/_results_R.csv"
# df_rma = pd.read_csv(rma_pred_delta, header=0)
# rma_array = df_rma.to_numpy()
# rma_res = tropo - rma_array

#%% plot predictions for rma
y_rma = delta_smooth

plt.figure(figsize=(12, 6))

# Plot original delta data
plt.subplot(1, 2, 1)
plt.imshow(y_rma, cmap='viridis', vmin = -0.12, vmax = -0.21, interpolation='nearest')
plt.colorbar()
plt.title('Original Delta (y_rma)')

# Plot RMA predictions
plt.subplot(1, 2, 2)
plt.imshow(rma_array, cmap='viridis', vmin = -0.12, vmax = -0.21, interpolation='nearest')
plt.colorbar()
plt.title('RMA Predictions')

plt.tight_layout()
plt.show()

# plot residual
fig = plt.figure()
ax = fig.add_subplot(121)
# Display the image
im = ax.imshow(smoothed_osl_res)
# Add a colorbar to the image
fig.colorbar(im, ax=ax)
plt.title('RMA Residual')
plt.show()

#%% plot residual with coastline
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Example data
# Replace these with your actual data arrays
lats = annual.lat.values  # Latitude array
# lats=np.flip(lats, axis=1)
# lats=np.flip(lats, axis=0)  #flip for plotting
# lats=np.flip(lats, axis=1)
lons = annual.lon.values  # Longitude array
# lons=np.flip(lons, axis=1)
# lons=np.flip(lons, axis=0)  #flip for plotting
# lons=np.flip(lons, axis=1)
Lons, Lats = np.meshgrid(lons, lats)  # Create a grid for the data

residual
residual=np.flip(residual, axis=1)
residual=np.flip(residual, axis=0)  #flip for plotting
residual=np.flip(residual, axis=1)

# Create the plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add geographical features
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, edgecolor='black')

# Plot the data
im = ax.pcolormesh(Lons, Lats, residual, shading='auto', cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.1)
cbar.set_label('Residual Value')

# Set the title
ax.set_title('RMA Residual')

plt.show()



#%% save as raster for RF

#assign spatial dimensions based on original nc file
aligned_xarr = xr.DataArray(residual, dims=summer.dims, coords=summer.coords)
aligned_xarr = aligned_xarr.rio.set_spatial_dims('lon','lat')

#save as raster
aligned_xarr.rio.to_raster(r"no_smoothing_winter.tif")


#clip


#%% load excel files with dataframe from residual (created in random forest code)


file = '/Users/catcoll/Documents/spatial_data/OLS_summer.xlsx'
data = pd.read_excel(file)

x = data['numerical_feature_0']
a = data['numerical_feature_1']
canopy = x.values
imper=a.values
#target column
y = data['residual']
res = y.values


plt.plot(a, res, 'o')






