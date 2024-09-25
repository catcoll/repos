#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:22:04 2024

Organzied zonal stats

@author: catcoll

This code calculates residual from attentuation model,
calculates distribution of residual values with in shapefiles,
then plots a box plot of all zones on the same plot
    
"""


import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
from shapely.ops import unary_union
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import seaborn as sns


#%% Import shape files

"""
Import shapefiles for desired areas to conduct zonal stats

"""

houston=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/houston.shp")
houston.crs #EPSG: 4326
#combine houston into 1
combined_geometry = unary_union(houston.geometry)
combined_gdf = gpd.GeoDataFrame({'geometry': [combined_geometry]}, crs=houston.crs)
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
# combined_gdf.to_file('combined_coral.shp')
coral = combined_coral

gulf_mex=gpd.read_file("/Users/catcoll/Documents/spatial_data/gulf_of_mex/gulf_reproj_4326.shp")
gulf_mex.crs #EPSG: 4326


non_urban=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/non_urban_dissolved.shp")
non_urban.crs #EPSG: 4326

shapefiles = [houston, new_o, tampa, coral, non_urban, gulf_mex]

#%% Calculate residual



#open model data and average over season
x = r'/Users/catcoll/Documents/py/isotopes/atten_tropomi_v2.nc'
model = xr.open_dataset(x)
model=model.sel(lat=slice(22,34), lon=slice(258,282))
model_del=model.delta_est
model_del_22=model_del.sel(time='2022')

model_s=model_del_22.groupby("time.season").mean()
model_s=model_s.sel(season=["DJF","MAM","JJA","SON"])


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

#convert to per mil
gulf_23_dw25_s=gulf_23_dw25_s*1000

#plot tropomi and model data
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


# #regrid to smaller grid if needed

# y=np.arange(22, 34, 0.01)
# x=np.arange(258,282, 0.01)

# residuals_re= residuals.interp(lon=x, lat=y)
# residuals_re

# ps25 = residuals_re.plot.pcolormesh(
#     col="season", 
#     transform=ccrs.PlateCarree(), cmap='Spectral_r',
#     #norm = mpl.colors.Normalize(vmin=-.25, vmax=-.05),
#     subplot_kws={'projection': ccrs.Orthographic(-80,45)}
#     )
# for ax in ps25.axs.flat:
#     ax.coastlines()
#     ax.gridlines()
# ax.set_extent([260,280,24,33])

#%%

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
combined_values = collect_seasonal_data(residuals, gulf_mex, transform)

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


labels = ["Houston", "New Orleans", "Tampa", "Cape Coral", "Non-Urban", "Open Ocean"]

shapefiles
data_dict = {}
for idx, gdf in enumerate(shapefiles):
    try:
        label = labels[idx]
        combined_values = collect_seasonal_data(residuals, gdf, transform) # call function to collect all data in region
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




