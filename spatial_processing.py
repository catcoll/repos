#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:46:53 2024

@author: catcoll

In this code:
    nothing
"""
import geopandas as gpd
import numpy as np 
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Point
import cartopy.crs as ccrs


#extract data for delta weighted only

ds=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc")
ds=ds.sel(lat=slice(21,34), lon=slice(260,280))
dd=ds.deltad_weighted  #extract delta values
lon=dd.lon.values
lat=dd.lat.values
ds.crs
#Convert to dataframe to convert to geodataframe
df=dd.to_dataframe()    #convert to dataframe
df=gpd.GeoDataFrame(df) 

#format to acheive geometry
df_reset = df.reset_index() #unpack variables so they each are assigned a column
geometry = [Point(lon, lat) for lon, lat in zip(df_reset['lon'], df_reset['lat'])] # Create geometry points using lon and lat columns
gdf = gpd.GeoDataFrame(df_reset, geometry=geometry, crs='EPSG:4269')
gdf.to_crs(4269)





#%%rasterize data?
#tutorial: https://corteva.github.io/geocube/stable/examples/rasterize_point_data.html


import json
from functools import partial

import geopandas
from shapely.geometry import box, mapping

from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial

gdf.head()
lat

#conver back to xarray
geo_grid = make_geocube(
    vector_data=gdf,
    measurements=['deltad_weighted'],
    resolution=(0.25, 0.25),
    rasterize_function=rasterize_points_griddata,
)

type(geo_grid)
geo_grid.deltad_weighted.where(geo_grid.deltad_weighted!=geo_grid.deltad_weighted.rio.nodata).plot()

#%% import raster data
#tutorial: https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/fundamentals-raster-data/open-lidar-raster-python/



# Import necessary packages
import os
import matplotlib.pyplot as plt

# Use geopandas for vector data and rasterio for raster data
import geopandas as gpd
import rasterio as rio
# Plotting extent is used to plot raster & vector data together
from rasterio.plot import plotting_extent
from rasterio.plot import show





#open from path and read in data with .read(1)
with rio.open("/Users/catcoll/Documents/spatial_data/clips/forest_clip.tif") as forest:
    forest_clip = forest.read(1)
forest.shape

forest.crs


forest_clip=rio.open("/Users/catcoll/Documents/spatial_data/clips/forest_clip.tif")
fig, ax = plt.subplots()
extent = [forest_clip.bounds[0], forest_clip.bounds[2], forest_clip.bounds[1], forest_clip.bounds[3]] #plot on real coords
ax = rio.plot.show(forest_clip, extent=extent, ax=ax, cmap="Greens")


gdf.plot(ax=ax)



#houston shape file
houston=gpd.read_file("/Users/catcoll/Documents/spatial data/houston_shape_file/tl_2019_48225_faces.shp")

houston.plot("TFID", legend=True)



new_o=gpd.read_file("/Users/catcoll/Documents/spatial data/new_orleans/geo_export_b8d20e12-37e5-48cd-8860-2a2e49f80054.shp")
new_o

new_o.plot("gnocdc_lab")

tampa=gpd.read_file("/Users/catcoll/Documents/spatial data/Neighborhoods_Tampa/Neighborhoods.shp")
tampa

tampa.plot("OBJECTID")


#need to look harder for better shape file
mobile=gpd.read_file("/Users/catcoll/Documents/spatial data/tl_2022_01097_edges/tl_2022_01097_edges.shp")
mobile

mobile.plot("STATEFP")
















