#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:46:53 2024

@author: catcoll
"""
import geopandas as gpd
import numpy as np 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Point


#extract data for delta weighted only

ds=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc")
ds=ds.sel(lat=slice(21,34), lon=slice(260,280))
dd=ds.deltad_weighted  #extract delta values
lon=dd.lon.values
lat=dd.lat.values

#Convert to dataframe to convert to geodataframe
df=dd.to_dataframe()    #convert to dataframe
df=gpd.GeoDataFrame(df) 

#format to acheive geometry
df_reset = df.reset_index() #unpack variables so they each are assigned a column
geometry = [Point(lon, lat) for lon, lat in zip(df_reset['lon'], df_reset['lat'])] # Create geometry points using lon and lat columns
gdf = gpd.GeoDataFrame(df_reset, geometry=geometry, crs='EPSG:4326')
gdf





gdf.plot(markersize=.01)  
plt.title('Spatial Distribution of deltad_weighted')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()