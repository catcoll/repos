#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:37:48 2024

uploading raster and nc data.
Averaging values within bounds based on metropolitan 
areas along the gulf (uploaded as shapefiles)


@author: catcoll
"""
from osgeo import gdal
import rasterio as rio
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, mapping
from rasterio.mask import mask
from rasterstats import zonal_stats
import numpy as np


# WANT TO MAKE SURE EVERYTHING IS IN THE SAME PROJECTION.!

#%%uploading metropolitan area shapefiles


houston=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/houston_again.shp")
houston #ESGP:4269

new_o=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/new_o_again.shp")
new_o.crs #ESGP:4269

tampa=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/tampa_again.shp")
tampa.crs #ESGP:4269

coral=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/cape_coral_met.shp")
coral.crs #ESGP:4269

#%% calculate percentage of land type in each zone

#percent occurence for land types in urban areas.

"""
steps in this cell:
    
upload raster data
define color map according to land type categories
funcion to calculate percentage of occurence from output dictionary
run for each zone

"""
# Define bounding box coordinates
xmin,xmax,ymin,ymax=-100,-70,24,32.9 # set the region

# Create a Polygon geometry
bbox_geom = Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

# Convert Polygon to GeoJSON-like dictionary
bbox_geojson = mapping(bbox_geom)

# # Open the raster file
with rio.open("/Users/catcoll/Documents/spatial_data/clips/LC/lc_crop_4269.tif") as lc:
#     # Mask the raster with the bounding box
    data, _ = mask(lc, [bbox_geojson], crop=True)

#     fig, ax = plt.subplots()
#     extent = [lc.bounds[0], lc.bounds[2], lc.bounds[1], lc.bounds[3]] #plot on real coords
#     ax = rio.plot.show(lc, extent=extent, ax=ax, cmap="Greens")


lc.crs #ESGP: 4269
#categories for land cover
cmap = {0: "Other", 11:"Open water", 12:"Perennial ice", 21:"Developed, open space", 22:"Developed, low intensity",
        23:"Developed, med intensity", 24:"Developed, high intensity", 31:"Barren land",
        41:"Deciduous forest", 42:"Evergreen forest", 43:"Mixed forest", 51:"Dwarf shrub",
        52:"Shrub/scrub", 71:"Grassland", 72:"Sedge", 73:"Lichens", 74:"Moss", 81:"Pasture/Hay",
        82:"Cultivated crops", 90:"Woody wetlands", 95:"Herbaceous Wetlands"}

#calculate percent from dictionary
def percentage(total_occurrences, land_types):
    percentages = {land_cover: (occurrences / total_occurrences) * 100 for land_cover, occurrences in land_types[0].items()}
    percentages = {land_cover: round(percentage, 2) for land_cover, percentage in percentages.items()}

    return percentages  

#run stats for each zone
land_types_h = zonal_stats(houston,'/Users/catcoll/Documents/spatial_data/clips/LC/lc_crop_4269.tif',
            categorical=True, category_map=cmap)

land_types_no = zonal_stats(new_o,'/Users/catcoll/Documents/spatial_data/clips/LC/lc_crop_4269.tif',
            categorical=True, category_map=cmap)

land_types_t = zonal_stats(tampa,'/Users/catcoll/Documents/spatial_data/clips/LC/lc_crop_4269.tif',
            categorical=True, category_map=cmap)

land_types_cc = zonal_stats(coral,'/Users/catcoll/Documents/spatial_data/clips/LC/lc_crop_4269.tif',
            categorical=True, category_map=cmap)


total_occurrences_h = sum(land_types_h[0].values())
total_occurrences_no = sum(land_types_no[0].values())
total_occurrences_t = sum(land_types_t[0].values())
total_occurrences_cc = sum(land_types_cc[0].values())

# Calculate the percentages

houston_lc=percentage(total_occurrences_h, land_types_h)
new_o_lc=percentage(total_occurrences_no, land_types_no)
tampa_lc=percentage(total_occurrences_t, land_types_t)
coral_lc=percentage(total_occurrences_cc, land_types_cc)   

houston_lc
new_o_lc
tampa_lc
coral_lc
    
    
#%% summarize imperviousness data


with rio.open("/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif") as imperv:
    # Mask the raster with the bounding box
    data, _ = mask(imperv, [bbox_geojson], crop=True)

    # fig, ax = plt.subplots()
    # extent = [imperv.bounds[0], imperv.bounds[2], imperv.bounds[1], imperv.bounds[3]] #plot on real coords
    # ax = rio.plot.show(imperv, extent=extent, ax=ax, cmap="Greens")
 
    
    
    
    
imperv_h = zonal_stats(houston,'/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])    
imperv_h
imperv_no = zonal_stats(new_o,'/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])    
imperv_no
imperv_t = zonal_stats(tampa,'/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])    
imperv_t
imperv_cc = zonal_stats(coral,'/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])    
imperv_cc

#%% summarize forest cover



with rio.open("/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif") as imperv:
    # Mask the raster with the bounding box
    data, _ = mask(imperv, [bbox_geojson], crop=True)

    # fig, ax = plt.subplots()
    # extent = [imperv.bounds[0], imperv.bounds[2], imperv.bounds[1], imperv.bounds[3]] #plot on real coords
    # ax = rio.plot.show(imperv, extent=extent, ax=ax, cmap="Greens")
 
forest_h = zonal_stats(houston,'/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])
forest_h
forest_no = zonal_stats(new_o,'/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])  
forest_no
forest_t = zonal_stats(tampa,'/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])  
forest_t
forest_cc = zonal_stats(coral,'/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])
forest_cc


#%% summarize nc data


import rasterio as rio
import geopandas as gpd
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.features import geometry_mask
from affine import Affine
# load and read shp-file with geopandas
houston=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/houston_again.shp")
# load and read netCDF-file to dataset and get datarray for variable

x = r'/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc'
ds = xr.open_dataset(x)
ds=ds.sel(lat=slice(21,34), lon=slice(260,280))
ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180 # convert to -180:180

delta=ds.deltad_weighted

annual_mean=delta.groupby("time.year").mean()
seasonal_mean=delta.groupby("time.season").mean()

ds0=annual_mean.sel(year=annual_mean["year"].values[1])
ds0

#vars 

lon=ds0.lon
lat=ds0.lat
lon_mesh, lat_mesh = np.meshgrid(lon, lat)

gdf=houston

# practice for just one year


# # Load NetCDF data


# latitude = ds0['lat'].values  
# longitude = ds0['lon'].values  



transform = Affine(1.0, 0.0, lon.min(), 0.0, -1.0, lat.max())
# # Create a mask from shapefile geometry
# mask = geometry_mask(gdf.geometry, out_shape=(len(latitude), len(longitude)), transform=transform, invert=True)

variable = ds0.values
masked_data = np.where(mask, variable, np.nan)

stats = zonal_stats(gdf, masked_data, affine=transform, stats=['mean', 'sum', 'count', 'min', 'max'])
#%% iterate over each year and collect stats
annual_mean['year']


#trying to iterate through each year
yearly_stats = []
yearly_stats.clear() #clear list if running loop again
# Iterate over each year in the NetCDF dataset
for year in range(8):
    
    variable = annual_mean.sel(year=annual_mean["year"].values[year]) # Extract data for the current year
    mask = geometry_mask(gdf.geometry, out_shape=(len(lat), len(lon)), transform=transform, invert=True) # Create a mask from shapefile geometry

    # Apply the mask to NetCDF data for the current year
    masked_data = np.where(mask, variable, np.nan)
    
    # Perform zonal statistics for the current year
    year_stats = zonal_stats(gdf, masked_data, affine=transform, stats=['mean', 'min', 'max'])
    
    # Append zonal statistics for the current year to the list
    yearly_stats.append(year_stats)

#zonal statistics for each year in the `yearly_stats` list



for year, stats in enumerate(yearly_stats):
    print(f"Statistics in for Year {annual_mean['year'].values[year]}:")
    for idx, stat in enumerate(stats):
        print(f"Zone {idx+1}: Mean = {stat['mean']}, Min = {stat['min']}, Max = {stat['max']}")
    print("\n")


#%% seasonal stats

seasonal_mean
seasonal_mean['season']


#trying to iterate through each year
seasonal_stats = []
seasonal_stats.clear() #clear list if running loop again
# Iterate over each year in the NetCDF dataset
for season in range(4):
    # Extract data for the current year
    variable = seasonal_mean.sel(season=seasonal_mean["season"].values[season])
    
    # Create a mask from shapefile geometry
    mask = geometry_mask(gdf.geometry, out_shape=(len(lat), len(lon)), transform=transform, invert=True)
    
    # Apply the mask to NetCDF data for the current year
    masked_data = np.where(mask, variable, np.nan)
    
    # Perform zonal statistics for the current season
    season_stats = zonal_stats(gdf, masked_data, affine=transform, stats=['mean', 'min', 'max'])
    
    # Append zonal statistics for the current year to the list
    seasonal_stats.append(season_stats)

#zonal statistics for each season in list



for season, stats in enumerate(seasonal_stats):
    print(f"Statistics for Season {seasonal_mean['season'].values[season]}:")
    for idx, stat in enumerate(stats):
        print(f"Zone {idx+1}: Mean = {stat['mean']},  Min = {stat['min']}, Max = {stat['max']}")
    print("\n")



