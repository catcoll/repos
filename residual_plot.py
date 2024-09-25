#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:25:40 2024

@author: catcoll

In this code:
    
    Uploads all shapefiles including individual land types
    Reclassifies raster, condenses similar land types into one (only need to make once)
    Calculates residual, regrids, and saves result as a raster (from attenuation model)
    Clips raster with each shapefile and creates a box plot with all zones
    Includes stats and plot for tukey kramer and bonferroni tests
    
"""



import xarray as xr
import rioxarray as rio
from shapely.geometry import mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import numpy as np
import pandas as pd
import rasterio

#%% load shape files and combine complex shapefiles into one shape, create list of all shapefiles

houston=gpd.read_file("/Users/catcoll/Documents/spatial_data/constrained_urban/houston.shp")
houston.crs #EPSG: 4326
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

all_urban=gpd.read_file('/Users/catcoll/Documents/spatial_data/constrained_urban/cropped_all.shp')

open_water = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_11_05.shp')
open_water.crs #EPSG: 4326
# combine cape coral into 1
combined_water = unary_union(open_water.geometry)
combined_water = gpd.GeoDataFrame({'geometry': [combined_water]}, crs='epsg:4326')
open_water=combined_water

urban_again = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_21_05.shp')
urban_again.crs #EPSG: 4326
# combine cape coral into 1
combined_urban_again = unary_union(urban_again.geometry)
combined_urban_again = gpd.GeoDataFrame({'geometry': [combined_urban_again]}, crs='epsg:4326')
urban_again=combined_urban_again

barren = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_31_05.shp')
barren.crs #EPSG: 4326
# combine cape coral into 1
combined_barren = unary_union(barren.geometry)
combined_barren = gpd.GeoDataFrame({'geometry': [combined_barren]}, crs='epsg:4326')
barren=combined_barren

forest = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_41_05.shp')
forest.crs #EPSG: 4326
# combine cape coral into 1
combined_forest = unary_union(forest.geometry)
combined_forest = gpd.GeoDataFrame({'geometry': [combined_forest]}, crs='epsg:4326')
forest=combined_forest

shrub = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_51_05.shp')
shrub.crs #EPSG: 4326
# combine cape coral into 1
combined_shrub = unary_union(shrub.geometry)
combined_shrub = gpd.GeoDataFrame({'geometry': [combined_shrub]}, crs='epsg:4326')
shrub=combined_shrub

moss = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_71_05.shp')
moss.crs #EPSG: 4326
# combine cape coral into 1
combined_moss = unary_union(moss.geometry)
combined_moss = gpd.GeoDataFrame({'geometry': [combined_moss]}, crs='epsg:4326')
moss=combined_moss

ag = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_81_05.shp')
ag.crs #EPSG: 4326
# combine cape coral into 1
combined_ag = unary_union(ag.geometry)
combined_ag = gpd.GeoDataFrame({'geometry': [combined_ag]}, crs='epsg:4326')
ag=combined_ag

wetlands = gpd.read_file('/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05/land_use_90_05.shp')
wetlands.crs #EPSG: 4326
# combine cape coral into 1
combined_wetlands = unary_union(wetlands.geometry)
combined_wetlands = gpd.GeoDataFrame({'geometry': [combined_wetlands]}, crs='epsg:4326')
wetlands=combined_wetlands

coast = gpd.read_file('/Users/catcoll/Documents/spatial_data/clips/binary_coast.shp')
combined_coast = unary_union(coast.geometry)
combined_coast = gpd.GeoDataFrame({'geometry': [combined_coast]}, crs='epsg:4326')
coast = combined_coast

shapefiles = [houston, new_o, tampa, coral, all_urban, 
              non_urban, gulf_mex, open_water, urban_again, 
              barren, forest, shrub, moss, ag, wetlands, coast]

#%% reclassify raster
"""
Condenses land use type by editing metadata

only need to run once as it saves the reclassified raster to a file

"""



# # Paths to your rasters
# input_raster_path = '/Users/catcoll/Documents/spatial_data/clips/land_use_clip.tif'
# output_raster_path = '/Users/catcoll/Documents/spatial_data/clips/reclassify_land_use_clip.tif'

# # Open the input raster
# with rasterio.open(input_raster_path) as src:
#     data = src.read(1)
#     meta = src.meta

# # Define the values to reclassify
# open_intensity_value = 21 
# low_intensity_value = 22  # Replace with the actual value for 'Developed, low intensity'
# med_intensity_value = 23 
# high_intensity_value = 24  # Replace with the actual value for 'Developed, high intensity'
# urban_value = 21

# deciduous = 41
# evergreen= 42
# mixed=43
# forest=41

# hay=81
# crops=82
# ag=81

# woody=90
# herb =95
# wetlands=90

# dwarf=51
# scrub=52
# shrub=51

# grass = 71
# sedge = 72
# lichen=73
# moss=74
# nonvascular=71

# # Reclassify values
# reclassified_data = np.where(np.isin(data, [open_intensity_value, low_intensity_value, 
#                                             med_intensity_value, high_intensity_value]), urban_value, data)

# reclassified_data = np.where(np.isin(data, [deciduous, evergreen, mixed]), forest, reclassified_data)
# reclassified_data = np.where(np.isin(data, [hay, crops]), ag, reclassified_data)
# reclassified_data = np.where(np.isin(data, [woody, herb]), wetlands, reclassified_data)
# reclassified_data = np.where(np.isin(data, [dwarf, scrub]), shrub, reclassified_data)
# reclassified_data = np.where(np.isin(data, [grass, sedge, lichen, moss]), nonvascular, reclassified_data)


# # Update metadata for the output raster
# meta.update(dtype=rasterio.uint8, count=1)

# # Write the reclassified raster
# with rasterio.open(output_raster_path, 'w', **meta) as dst:
#     dst.write(reclassified_data, 1)






#%% calculate

"""
This block calculates the residual from attenuation model, regrids if needed and 
saves result as a raster.

Bottom of this block loads the saves raster files

"""
# #open tropomi data and average
# y = r'/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc'
# ds25=xr.open_dataset(y)
# dw25=ds25.deltad_weighted
# gulf25=dw25.sel(lat=slice(25,32), lon=slice(258,282))
# gulf_23_dw25=gulf25.sel(time='2022')
# gulf_23_dw25_s=gulf_23_dw25.groupby("time.year").mean()
# gulf_23_dw25_s

# #convert to per mil
# gulf_23_dw25_s=gulf_23_dw25_s*1000

# #open model data
# x = r'/Users/catcoll/Documents/py/isotopes/atten_tropomi_v2.nc'
# model = xr.open_dataset(x)
# model=model.sel(lat=slice(25,32), lon=slice(258,282))
# model_del=model.delta_est
# model_del_22=model_del.sel(time='2022')
# model_s=model_del_22.groupby("time.year").mean()

# #calculate residual
# residual=gulf_23_dw25_s-model_s
# residual =residual.interpolate_na("lat")
# residuals =residual

# y=np.arange(25, 32, 0.05)
# x=np.arange(258,282, 0.05)

# #regrid if needed
# residuals_re= residuals.interp(lon=x, lat=y)
# residuals_re

# residuals.coords['lon'] = (residuals.coords['lon'] + 180) % 360 - 180  # Convert to -180:180
# residuals = residuals.sortby(residuals.lon)
# residuals = residuals.rio.set_spatial_dims('lon','lat')

# # #regrid to smaller grid if needed
# residuals_re.coords['lon'] = (residuals_re.coords['lon'] + 180) % 360 - 180  # Convert to -180:180
# residuals_re = residuals_re.sortby(residuals_re.lon)
# residuals_re = residuals_re.rio.set_spatial_dims('lon','lat')


# #save as a raster
# residuals.rio.to_raster(r"residual_raster_25.tif")
# residuals_re.rio.to_raster(r"residual_raster_05.tif")



#%% plots distribution of residual values in each shapefile

shapefiles
#open raster and set crs
raster = rio.open_rasterio('/Users/catcoll/Documents/py/isotopes/test.tif')
raster.rio.write_crs('epsg:4326', inplace=True)

all_values = []

#labels and colors for plotting
labels = ['Houston', "New Orleans", 'Tampa', 'Cape Coral', 'Urban', 
          'Non Urban', 'Open Ocean', 'Open Water', 'Urban Again', 'Barren Land',
          'Forest', 'Shrub', 'Moss', 'Agriculture', 'Wetlands', 'Near Coast']

colors = ['ivory','ivory','ivory', 'ivory', 'dimgrey',
          'salmon', 'navy', 'powderblue', 'slategrey', 'darkgoldenrod',
          'forestgreen', 'khaki', 'lime', 'peru', 'lightseagreen', 'blue']

data_frame = {}

#clip raster with shape geometry
for shape in shapefiles:
    clip = raster.rio.clip(shape.geometry.apply(mapping), shape.crs)
    
    values=clip.values.flatten()
    values=values[~np.isnan(values)]
    print(values.mean())
    all_values.append(values)



#plot box plot of all regions
plt.figure(figsize=(12, 6))
bplot=plt.boxplot(all_values, 
                  patch_artist =True,
                  notch = True,
                  labels=labels)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
# for median in bplot['medians']:
#     median.set(color ='red',
#                linewidth = 3)
for flier in bplot['fliers']:
    flier.set(marker ='o',
              color ='#e7298a',
              alpha = 0.5)
plt.xticks(rotation=45)
plt.title("Residual Value Distributions by Land Characteristic, high pass filter summer")
plt.ylabel("Residual (per mil)")
plt.show()


#%% statistical tests: tukey, bonferroni
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
import itertools

data_frame = {label: values for label, values in zip(labels, all_values)}
#to dataframe
df = pd.DataFrame([(key, val) for key, arr in data_frame.items() for val in arr], columns=['Group', 'Value'])


model = ols('Value ~ C(Group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)



# Perform Tukey's HSD
tukey = pairwise_tukeyhsd(endog=df['Value'], groups=df['Group'], alpha=0.05)
print(tukey)
tukey_df = tukey.summary().data[1:]
tukey_df = pd.DataFrame(tukey_df, columns=tukey.summary().data[0])

# Plot Tukey's HSD results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tukey_df['meandiff'], y=tukey_df['reject'], hue=tukey_df['reject'], palette={True: 'red', False: 'blue'}, s=100)

# Add labels and titles
plt.axvline(x=0, color='grey', linestyle='--', linewidth=1)
plt.xlabel('Mean Difference')
plt.ylabel('Reject Null Hypothesis (True/False)')
plt.title('Tukey\'s HSD Test Results')
plt.yticks([True, False], ['Reject Null Hypothesis', 'Fail to Reject Null Hypothesis'])
plt.grid(True)
plt.show()


# t test
group1 = data_frame['Urban']
group2 = data_frame['Forest']


# Perform independent t-test
t_stat, p_value = ttest_ind(group1, group2)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


#bonferroni

# Create a list of all pairwise group comparisons
groups = df['Group'].unique()
comparisons = list(itertools.combinations(groups, 2))

# Store results
results = []

# Perform pairwise t-tests
for group1, group2 in comparisons:
    values1 = df[df['Group'] == group1]['Value']
    values2 = df[df['Group'] == group2]['Value']
    
    # Perform t-test
    t_stat, p_value = ttest_ind(values1, values2, equal_var=False)  # Use Welch's t-test if variances are unequal
    
    # Store results
    results.append({
        'Group1': group1,
        'Group2': group2,
        't-statistic': t_stat,
        'p-value': p_value
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Apply Bonferroni correction
results_df['Bonferroni p-value'] = results_df['p-value'] * len(comparisons)
results_df['Bonferroni p-value'] = results_df['Bonferroni p-value'].clip(upper=1.0)  # p-value cannot exceed 1.0

print(results_df)


import matplotlib.pyplot as plt


# Create the plot
plt.figure(figsize=(12, 8))
# Set threshold for significance
alpha = 0.05
# Store handles for legend
handles = []
# Plot all comparisons
for index, row in results_df.iterrows():
    # Plot all comparisons in grey
    plt.plot(index, row['Bonferroni p-value'], 'o', color='grey', alpha=0.6)
    plt.text(index, row['Bonferroni p-value'], f'{row["Bonferroni p-value"]:.3f}', verticalalignment='bottom', horizontalalignment='right', fontsize=9, color='grey')

    # If the comparison is significant, add it to the legend
    if row['Bonferroni p-value'] > alpha:
        handle, = plt.plot(index, row['Bonferroni p-value'], 'o', label=f'{row["Group1"]} vs {row["Group2"]}')
        handles.append(handle)

# Add a horizontal line for the alpha level
plt.axhline(y=alpha, color='red', linestyle='--', label=f'Alpha = {alpha}')

# Add title and labels
plt.ylim(0, 1.1)
plt.ylabel('Bonferroni Corrected p-value')
plt.title('Pairwise Comparisons with Bonferroni Correction')
plt.xticks([])

# Only show significant comparisons in the legend
plt.legend(handles=handles, loc='best')
plt.grid(True)
plt.show()




