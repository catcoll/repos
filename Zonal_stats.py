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
import cartopy.crs as ccrs
import matplotlib as mpl
# WANT TO MAKE SURE EVERYTHING IS IN THE SAME PROJECTION.!

#%%uploading metropolitan area shapefiles
from shapely.ops import unary_union

houston=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/houston.shp")
houston.crs #EPSG: 4326

#combine houston into 1
combined_geometry = unary_union(houston.geometry)
# Create a new GeoDataFrame with the combined geometry
combined_gdf = gpd.GeoDataFrame({'geometry': [combined_geometry]}, crs=houston.crs)
# Save the combined GeoDataFrame to a new shapefile
combined_gdf.to_file('combined_houston.shp')
houston = combined_gdf

new_o=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/new_orleans.shp")
new_o.crs #EPSG: 4326

tampa=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/test.shp")
tampa.crs #EPSG: 4326

coral=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/cape_coral.shp")
coral.crs #EPSG: 4326
# combine cape coral into 1
combined_coral = unary_union(coral.geometry)
combined_coral = gpd.GeoDataFrame({'geometry': [combined_coral]}, crs=coral.crs)
combined_gdf.to_file('combined_coral.shp')
coral = combined_coral


non_urban=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/non_urban_dissolved.shp")
non_urban.crs #EPSG: 4326

all_urban=gpd.read_file('/Users/catcoll/Documents/spatial_data/constrained_urban/cropped_all.shp')
combined_all = unary_union(all_urban.geometry)
combined_all = gpd.GeoDataFrame({'geometry': [combined_all]}, crs=coral.crs)
combined_gdf.to_file('combined_urban.shp')
all_urban = combined_all




#larger shape file that work
houston1=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/houston_again.shp")
tampa1=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/tampa_again.shp")
new_o1=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/new_o_again.shp")
coral1=gpd.read_file("/Users/catcoll/Documents/spatial_data/metropolitan_areas/better_files/cape_coral.shp")





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

land_types_non = zonal_stats(non_urban,'/Users/catcoll/Documents/spatial_data/clips/LC/lc_crop_4269.tif',
            categorical=True, category_map=cmap)
land_types_all = zonal_stats(all_urban,'/Users/catcoll/Documents/spatial_data/clips/LC/lc_crop_4269.tif',
            categorical=True, category_map=cmap)


total_occurrences_h = sum(land_types_h[0].values())
total_occurrences_no = sum(land_types_no[0].values())
total_occurrences_t = sum(land_types_t[0].values())
total_occurrences_cc = sum(land_types_cc[0].values())
total_occurrences_non = sum(land_types_non[0].values())
total_occurrences_all = sum(land_types_all[0].values())

# Calculate the percentages

houston_lc=percentage(total_occurrences_h, land_types_h)
new_o_lc=percentage(total_occurrences_no, land_types_no)
tampa_lc=percentage(total_occurrences_t, land_types_t)
coral_lc=percentage(total_occurrences_cc, land_types_cc)  
non_lc=percentage(total_occurrences_non, land_types_non)   
all_lc=percentage(total_occurrences_all, land_types_all)   

houston_lc
new_o_lc
tampa_lc
coral_lc
non_lc
all_lc
    
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
imperv_non = zonal_stats(non_urban,'/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])    
imperv_non
imperv_all = zonal_stats(all_urban,'/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])    
imperv_all


#%% summarize forest cover



with rio.open("/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif") as forest:
    # Mask the raster with the bounding box
    data, _ = mask(forest, [bbox_geojson], crop=True)

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
forest_non = zonal_stats(non_urban,'/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])
forest_non
forest_all = zonal_stats(all_urban,'/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif', stats=['min', 'max', 'mean', 'majority', 'std', 'range'])
forest_all

#%% Summarize nc data MUST RUN THIS CELL


import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterstats import zonal_stats

# Load the NetCDF dataset
x = r'/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc'
ds = xr.open_dataset(x)

# Select region of interest
ds = ds.sel(lat=slice(21,34), lon=slice(260,280))
ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180  # Convert to -180:180

# Compute annual and seasonal means
delta = ds.deltad_weighted
annual_mean = delta.groupby("time.year").mean()
seasonal_mean = delta.groupby("time.season").mean()

# Select a specific year for analysis
ds0 = annual_mean.sel(year=annual_mean["year"].values[1])

# Extract variables
lon = ds0.lon.values
lat = ds0.lat.values
variable = ds0.values

# Create a meshgrid for the data
lon_mesh, lat_mesh = np.meshgrid(lon, lat)

# Load the shapefile and ensure it's in the same CRS as the raster
gdf = coral
gdf = gdf.to_crs(epsg=4326)  # Assuming WGS84, adjust if necessary

# Create affine transform based on data resolution and extent
transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

# Create a mask from shapefile geometry
mask = geometry_mask(gdf.geometry, transform=transform, invert=True, out_shape=variable.shape)

# Mask the data
masked_data = np.where(mask, variable, np.nan)

# Calculate zonal statistics
stats = zonal_stats(gdf, masked_data, affine=transform, stats=['mean', 'sum', 'count', 'min', 'max'])

print(stats)



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



#%% model data




#open model data and average over season
x = r'/Users/catcoll/Documents/py/isotopes/atten_tropomi_v2.nc'
model = xr.open_dataset(x)
model=model.sel(lat=slice(22,34), lon=slice(258,282))
model
#open tropomi data and average over season
y = r'/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc'
ds25=xr.open_dataset(y)
dw25=ds25.deltad_weighted
dw25

gulf25=dw25.sel(lat=slice(22,34), lon=slice(258,282))
gulf_23_dw25=gulf25.sel(time='2022')
gulf_23_dw25_s=gulf_23_dw25.groupby("time.season").mean()
gulf_23_dw25_s=gulf_23_dw25_s.sel(season=["DJF","MAM","JJA","SON"])
gulf_23_dw25_s

#interpolate

# t_lat=gulf_23_dw25_s.lat
# t_lon=gulf_23_dw25_s.lon
# model_regrid=model.interp(lat=t_lat, lon=t_lon)
# model_regrid

#average over season for model data
model_del=model.delta_est
model_del_22=model_del.sel(time='2022')

model_s=model_del_22.groupby("time.season").mean()
model_s=model_s.sel(season=["DJF","MAM","JJA","SON"])

#convert to tropomi units
gulf_23_dw25_s=gulf_23_dw25_s*1000

#plot model data
ps25 = gulf_23_dw25_s.plot.pcolormesh(
    col="season", 
    transform=ccrs.PlateCarree(), cmap='Spectral_r',
    # norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps25.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,24,33])
plt.savefig('/Users/catcoll/Documents/figs/seasonal_25km_2022.png', dpi=300, bbox_inches='tight')

ps25 = model_s.plot.pcolormesh(
    col="season", 
    transform=ccrs.PlateCarree(), cmap='Spectral_r',
    # norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps25.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,24,33])
plt.savefig('/Users/catcoll/Documents/figs/model_reverse_high_qual.png', dpi=300, bbox_inches='tight')

#calculate residual

residual=gulf_23_dw25_s-model_s
residual =residual.interpolate_na("lat")
residuals =residual
residuals

#plot residual
ps25 = residuals.plot.pcolormesh(
    col="season", 
    transform=ccrs.PlateCarree(), cmap='Spectral_r',
    #norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps25.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,24,33])
plt.savefig('/Users/catcoll/Documents/figs/residual_high_qual.png', dpi=300, bbox_inches='tight')



residual
#regrid to smaller grid

y=np.arange(22, 34, 0.01)
x=np.arange(258,282, 0.01)

residuals_re= residuals.interp(lon=x, lat=y)
residuals_re

ps25 = residuals_re.plot.pcolormesh(
    col="season", 
    transform=ccrs.PlateCarree(), cmap='Spectral_r',
    #norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps25.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,24,33])

#%% annual average

model_del_22


model_a=model_del_22.mean(dim='time')
gulf_23_dw25_a=gulf_23_dw25.mean(dim='time')

residual_a = gulf_23_dw25_a - model_a
model_a

ps25 = model_a.plot.pcolormesh(
    transform=ccrs.PlateCarree(), cmap='Spectral',
    #norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
    # subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps25.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,24,33])


ps25 = gulf_23_dw25_a.plot.pcolormesh(
    col="season", 
    transform=ccrs.PlateCarree(), cmap='Spectral',
    #norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps25.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,24,33])


ps25 = residual_a.plot.pcolormesh( 
    transform=ccrs.PlateCarree(), cmap='Spectral',
    #norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
    subplot_kws={'projection': ccrs.Orthographic(-80,45)}
    )
for ax in ps25.axs.flat:
    ax.coastlines()
    ax.gridlines()
ax.set_extent([260,280,24,33])






#%% box plots!! this cell just creates one box plot, you must select only 1 time var


#shape files needed:
    
houston
new_o
tampa
coral
gulf_mex=gpd.read_file("/Users/catcoll/Documents/spatial_data/gulf_of_mex/gulf_reproj_4326.shp")
gulf_mex.crs #EPSG: 4326
gulf_mex.plot()
gulf_mex

non_urban=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/non_urban_dissolved.shp")
non_urban.crs #EPSG: 4326
non_urban.plot()

shapefiles = [houston, new_o, tampa, coral, non_urban, gulf_mex]


import rasterio as rio
import geopandas as gpd
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.features import geometry_mask
from affine import Affine
import pandas as pd

# nc file to use, already has just one var
residuals_re.coords['lon'] = (residuals_re.coords['lon'] + 180) % 360 - 180 # convert to -180:180

residual
residual=residuals_re.sel(season=residuals["season"].values[0])

#vars 

lon_r=residual.lon
lat_r=residual.lat
lon_mesh, lat_mesh = np.meshgrid(lon_r, lat_r)

gdf=houston
houston
latitude = residual['lat'].values  
longitude = residual['lon'].values  

variable = residual.values

transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

# Create a mask from shapefile geometry
mask = geometry_mask(gdf.geometry, transform=transform, invert=False, out_shape=variable.shape)



# Mask the data
masked_data = np.where(mask, variable, np.nan)



stats = zonal_stats(houston, masked_data, affine=transform, stats=['mean','range'])
stats


def extract_values_for_zones(zones, data, transform):
    zone_values = []
    
    for _, zone in zones.iterrows():
        # Mask the data with the current zone
        zone_mask = geometry_mask([zone.geometry], out_shape=data.shape, transform=transform, invert=True)
        zone_data = np.where(zone_mask, data, np.nan)
        
        # Flatten and filter out NaN values
        zone_values.append(zone_data[~np.isnan(zone_data)])
    
    return zone_values

# Extract values for each zone
zone_values = extract_values_for_zones(houston, masked_data, transform)
zone_values
# Flatten the list of values
all_values = np.concatenate(zone_values)

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(all_values)
plt.ylabel('Raster Values')
plt.title('Box Plot of Raster Values for All Zones')
plt.show()

#%% plot each season

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import xarray as xr

# Define the function to process each season
def process_season(season, residual, gdf, transform):
    # Select data for the current season
    seasonal_data = residual.sel(season=season)
    
    # Extract variable values
    variable = seasonal_data.values

    # Create a mask from shapefile geometry
    mask = geometry_mask(gdf.geometry, out_shape=(len(seasonal_data['lat']), len(seasonal_data['lon'])), transform=transform, invert=True)
    
    # Mask the data
    masked_data = np.where(mask, variable, np.nan)
    
    # Extract individual raster values for each zone
    def extract_values_for_zones(zones, data, transform):
        zone_values = []
        
        for _, zone in zones.iterrows():
            # Mask the data with the current zone
            zone_mask = geometry_mask([zone.geometry], out_shape=data.shape, transform=transform, invert=True)
            zone_data = np.where(zone_mask, data, np.nan)
            
            # Flatten and filter out NaN values
            zone_values.append(zone_data[~np.isnan(zone_data)])
        
        return zone_values
    
    # Extract values for each zone
    zone_values = extract_values_for_zones(gdf, masked_data, transform)
    
    # Flatten the list of values
    all_values = np.concatenate(zone_values)
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_values)
    plt.ylabel('Raster Values')
    plt.title(f'Box Plot of Raster Values for Season {season}')
    plt.show()

# Load your raster data (example NetCDF file)

x = r'/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc'
residuals = xr.open_dataset(x).deltad_weighted

# Example: Select a subset of data (if needed)
residuals = residuals.sel(lat=slice(21,34), lon=slice(260,280))
residuals.coords['lon'] = (residuals.coords['lon'] + 180) % 360 - 180  # Convert to -180:180

residuals = residuals.groupby("time.season").mean()


# Extract coordinates
lon_r = residuals.lon.values
lat_r = residuals.lat.values

# Create a mesh grid
lon_mesh, lat_mesh = np.meshgrid(lon_r, lat_r)

# Define the affine transform based on data resolution and extent
transform = from_origin(lon_r.min(), lat_r.max(), abs(lon_r[1] - lon_r[0]), abs(lat_r[1] - lat_r[0]))

# Load the shapefile as a GeoDataFrame
gdf = gulf_mex

# Ensure CRS match (assuming WGS84 for this example)
gdf = gdf.to_crs(epsg=4326)

# Get the list of unique seasons (assuming these are strings in your dataset)
seasons = residuals['season'].values

# Loop through each season and process
for season in seasons:
    process_season(season, residuals, gdf, transform)

zone_values

#%% plot combine all seasons

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from rasterio.features import geometry_mask
import xarray as xr


def collect_seasonal_data(residual, gdf, transform):
    all_seasonal_values = []

    def extract_values_for_zones(zones, data, transform):
        zone_values = []
        
        for _, zone in zones.iterrows():
            # Mask the data with the current zone
            zone_mask = geometry_mask([zone.geometry], out_shape=data.shape, transform=transform, invert=True)
            zone_data = np.where(zone_mask, data, np.nan)
            
            # Flatten and filter out NaN values
            zone_values.append(zone_data[~np.isnan(zone_data)])
        
        return zone_values

    # Get the list of unique seasons
    seasons = [0,1,2,3]

    # Iterate through each season
    for season in seasons:
        try:
            # Select data for the current season
            seasonal_data=residual.sel(season=residual["season"].values[season])
            
            # Ensure that the data is valid and not empty
            if seasonal_data.size == 0:
                print(f"No data for season {season}. Skipping.")
                continue

            # Extract variable values
            variable = seasonal_data.values

            # Create a mask from shapefile geometry
            mask = geometry_mask(gdf.geometry, out_shape=(len(seasonal_data['lat']), len(seasonal_data['lon'])), transform=transform, invert=True)
            
            # Mask the data
            masked_data = np.where(mask, variable, np.nan)
            
            # Extract values for each zone
            zone_values = extract_values_for_zones(gdf, masked_data, transform)
            
            # Flatten the list of values and append to the main list
            all_values = np.concatenate(zone_values)
            all_seasonal_values.append(all_values)

        except Exception as e:
            print(f"Error processing season {season}: {e}")

    # Combine all seasonal data
    combined_values = np.concatenate(all_seasonal_values) if all_seasonal_values else np.array([])

    return combined_values

def plot_boxplot(values, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(values)
    plt.ylabel('Raster Values')
    plt.title(title)
    plt.show()

# Load your raster data

residuals
# Extract coordinates
latitude = residuals['lat'].values  
longitude = residuals['lon'].values  
lon_r = residuals.lon.values
lat_r = residuals.lat.values

# Create a mesh grid
lon_mesh, lat_mesh = np.meshgrid(lon_r, lat_r)

# Define the affine transform
transform = Affine(1.0, 0.0, lon_r.min(), 0.0, -1.0, lat_r.max())

# Load the shapefile as a GeoDataFrame
shapefiles = [ houston, new_o, tampa, coral, non_urban, gulf_mex]

# Collect and aggregate data across all seasons
combined_values = collect_seasonal_data(residuals_re, houston, transform)

combined_values

# Plot the box plot of combined values
if combined_values.size > 0:
    plot_boxplot(combined_values, "HOUSTON")
else:
    print("No data available to plot.")


import matplotlib.pyplot as plt
import seaborn as sns  # For improved aesthetics

def plot_boxplot(values, title):
    plt.figure(figsize=(12, 8))
    
    # Set the style of the plot
    sns.set(style="whitegrid")
    
    # Create the box plot with additional customizations
    plt.boxplot(values, 
                patch_artist=True,  # Fill boxes with color
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(markerfacecolor='red', marker='o'),  # Customize outlier markers
                meanline=True, showmeans=True)# Add mean line


    # Add labels and title
    plt.ylabel('Raster Values', fontsize=14)
    plt.xlabel('Seasons', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add x-axis and y-axis ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Show the plot
    plt.show()

# Example usage
custom_title = 'Enhanced Box Plot of Raster Values Across All Seasons'
plot_boxplot(combined_values, custom_title)







#%%
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import xarray as xr
import seaborn as sns

# Define the function to process each season
def process_season(season, residual, gdf, transform):
    # Select data for the current season
    seasonal_data = residual.sel(season=season)
    
    # Ensure that the data is valid and not empty
    if seasonal_data.size == 0:
        print(f"No data for season {season}. Skipping.")
        return None

    # Extract variable values
    variable = seasonal_data.values

    # Create a mask from shapefile geometry
    mask = geometry_mask(gdf.geometry, out_shape=(len(seasonal_data['lat']), len(seasonal_data['lon'])), transform=transform, invert=True)
    
    # Mask the data
    masked_data = np.where(mask, variable, np.nan)
    cleaned_data = masked_data[~np.isnan(masked_data)]
    
    return cleaned_data
    # Extract values for each zone

process_season(residual['season'].values[0], residuals_re, houston, transform)
residual['season'].values

def collect_seasonal_data(residual, gdf, transform):
    """
    Collect and combine data across all seasons into a single array.
    """
    all_seasonal_values = []

    # Get the list of unique seasons
    seasons = residual['season'].values  # Ensure this matches your actual season values

    # Iterate through each season
    for season in seasons:
        try:
            seasonal_values = process_season(season, residual, gdf, transform)
            if seasonal_values is not None:
                all_seasonal_values.append(seasonal_values)
        except Exception as e:
            print(f"Error processing season {season}: {e}")

    # Combine all seasonal data into a single array
    combined_values = np.concatenate(all_seasonal_values) if all_seasonal_values else np.array([])
    
    # Print or plot to debug
    if combined_values.size > 0:
        print(f"Combined values - min: {np.min(combined_values)}, max: {np.max(combined_values)}")
    
    return combined_values    


def plot_boxplot(values, title):
    plt.figure(figsize=(12, 8))
    
    # Set the style of the plot
    sns.set(style="whitegrid")
    
    # Create the box plot with additional customizations
    plt.boxplot(values, 
                patch_artist=True,  # Fill boxes with color
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(markerfacecolor='red', marker='o'),  # Customize outlier markers
                meanline=True, showmeans=True)  # Add mean line

    # Add labels and title
    plt.ylabel('Raster Values', fontsize=14)
    plt.xlabel('Seasons', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add x-axis and y-axis ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Show the plot
    plt.show()

# Load your raster data
residuals 

# # (run if needed), do not rerun
# residuals = residuals.sel(lat=slice(21,34), lon=slice(260,280))
residuals.coords['lon'] = (residuals.coords['lon'] + 180) % 360 - 180  # Convert to -180:180

# Extract coordinates
lon_r = residuals.lon.values
lat_r = residuals.lat.values

# Create a mesh grid
lon_mesh, lat_mesh = np.meshgrid(lon_r, lat_r)

# Define the affine transform based on data resolution and extent
transform = from_origin(lon_r.min(), lat_r.max(), abs(lon_r[1] - lon_r[0]), abs(lat_r[1] - lat_r[0]))

# Load the shapefiles as GeoDataFrames
houston 
# Load other shapefiles similarly and combine if necessary

# Collect and aggregate data across all seasons
combined_values = collect_seasonal_data(residuals_re, gulf_mex, transform)

# Plot the box plot of combined values
if combined_values.size > 0:
    plot_boxplot(combined_values, "Non Urban")
else:
    print("No data available to plot.")

#%% plot all regions on same plot

def plot_multiple_boxplots(data_dict):
    plt.figure(figsize=(16, 8))
    sns.set(style="whitegrid")
    colors = sns.color_palette("husl", len(data_dict))


    for idx, (label, values) in enumerate(data_dict.items()):
        plt.boxplot(values, 
                    positions=[idx + 1],  # Position each box plot
                    patch_artist=True,  # Fill boxes with color
                    boxprops=dict(facecolor=colors[idx], color=colors[idx]),
                    whiskerprops=dict(color=colors[idx]),
                    capprops=dict(color=colors[idx]),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(markerfacecolor='red', marker='o'),
                    meanline=True, showmeans=True,
                    # showcaps=True,
                    widths=0.5)

    plt.ylabel('Residual', fontsize=25)
    
    # plt.title('Box Plot of Raster Values for Multiple GeoDataFrames', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(ticks=range(1, len(data_dict) + 1), labels=[f"{label[:10]}..." for label in data_dict.keys()], fontsize=18)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.show()


shapefiles1 = [houston1, new_o1, tampa1, coral1, non_urban, gulf_mex]
labels = ["Houston", "New Orleans", "Tampa", "Cape Coral", "Non-Urban", "Open Ocean"]

shapefiles
data_dict = {}
for idx, gdf in enumerate(shapefiles):
    try:
        label = labels[idx]
        combined_values = collect_seasonal_data(residuals_re, gdf, transform) # call function to collect all data in region
        if combined_values.size > 0:
            data_dict[label] = combined_values
        else:
            print(f"No data available for {label}. Skipping.")
    except Exception as e:
        print(f"Error processing GeoDataFrame {gdf}: {e}")

# Plot all box plots on the same figure after collecting all data
if data_dict:
    plot_multiple_boxplots(data_dict)
else:
    print("No data available to plot.")



data_dict

