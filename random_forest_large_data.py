#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:17:18 2024

@author: catcoll

In this code:
    
    Collect data for random forest
    resamples rasters to match reference raster (makes them the same resolution and shape)
    Extracts values from each pixel
    one-hot encodes categorical raster
    creates data frame for random forest model (can be used for other models)
    
    creates resampled rasters and saves to file
    cleans data/remove nan values
    
    trains model
    evaluate model
    plot feature importance
    
"""
import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import rioxarray as rio
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import mapping


#%% resample rasters function

#Function to resample raster to Tropomi data resolution (makes rasters to be same resolution)

def resample_raster(input_file, output_file, reference_file, method): # can be used with categorical or numerical raster, adjust method

    '''
    input file: file to resample/regrid (raster)
    output file: where you want to save the regridded raster to (string)
    reference_file: raster with target resolution (raster)
    method: bilinear for numerical, nearest for categorical
    '''
    
    try:
        # Open the reference file and get its properties (shape, transform, grid cell size)
        reference = gdal.Open(reference_file, gdal.GA_ReadOnly)
        reference_proj = reference.GetProjection()
        reference_trans = reference.GetGeoTransform()
        reference_extent = (
            reference_trans[0],  #top left x
            reference_trans[1],  #pixel width
            reference_trans[2],  #rotation (0 if North up)
            reference_trans[3],  #top left y
            reference_trans[4],  #rotation (0 if North up)
            reference_trans[5]   #pixel height (negative if North up)
        )
        
        #calculate the output bounds from the reference extent
        x_min = reference_extent[0]
        y_max = reference_extent[3]
        x_max = x_min + reference_extent[1] * reference.RasterXSize
        y_min = y_max + reference_extent[5] * reference.RasterYSize


        #specify parameters for resampling
        kwargs = {
            "format": "GTiff",
            "xRes": reference_trans[1],
            "yRes": abs(reference_trans[5]),
            "outputBounds": (x_min, y_min, x_max, y_max), #based on reference file extent
            "resampleAlg": method,  # use nearest for categorical
            "dstSRS": reference_proj # make sure rasters are in the same projection
        }
        
        # Perform the resampling using gdal.Warp
        gdal.Warp(output_file, input_file, **kwargs)
        print("Successfully saved")
        
    except Exception as e:
        print(f"Failed: {e}")


#%% Extract Raster Values Function

#extract and store values of each pixel, maintain identical shape/extent, indicate if not
def extract_values(raster_paths):
    values = []
    shapes = []
    
    # Iterate over file paths
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            band = src.read(1)
            shapes.append(band.shape) #shape
            values.append(band)  # raster values by pixel
    
    # ensure all rasters have the same shape
    if len(set(shapes)) != 1: # if shape is off, raise error
        raise ValueError('All rasters must have the same shape.')

    # stack all values into the same array
    stacked_values = np.stack(values, axis=-1)
    return stacked_values

#%% One-Hot Encoding Function

#create a feature for each land use type
def one_hot_encode(categorical_raster_path):
    with rasterio.open(categorical_raster_path) as src:
        data = src.read(1)
        #find all unique classes
        unique_classes = np.unique(data)  # pulls out all unique values in the data set
        
        #create one-hot encoded array
        one_hot_encoded = np.zeros((data.size, len(unique_classes)))
        for i, land_class in enumerate(unique_classes):
            #creates one hot encoded array for each class
            one_hot_encoded[:, i] = (data.flatten() == land_class).astype(int)
    
    return one_hot_encoded, unique_classes

#%% Prepare Data Function (Create Data Frame for Training Fxn)

#configure dataframe to perform random forest regression
def prepare_data(numerical_raster_paths, categorical_raster_path, residual_raster_path):
    # call one-hot fxn to encode the categorical raster
    one_hot_encoded, _ = one_hot_encode(categorical_raster_path)
    
    # call extract fxn for numerical values
    numerical_values = extract_values(numerical_raster_paths)
    numerical_values = numerical_values.reshape((-1, numerical_values.shape[-1]))
    
    # combine numerical and categorical features into same array/stack
    feature_values = np.hstack([numerical_values, one_hot_encoded])
    
    #keep track of name of each feature
    numerical_feature_names = [f'numerical_feature_{i}' for i in range(numerical_values.shape[1])]
    categorical_feature_names = [f'one_hot_{class_name}' for class_name in unique_classes]
    all_feature_names = numerical_feature_names + categorical_feature_names
    
    #read the residual raster values (this is target column for RF)
    with rasterio.open(residual_raster_path) as src:
        residual_values = src.read(1).flatten()
    
    #create DataFrame
    df = pd.DataFrame(feature_values, columns=all_feature_names)
    #add residual column
    df['residual'] = residual_values
    
    return df


#%% Load rasters
imper = "/Users/catcoll/Documents/spatial_data/clips/imper_clip_no_data_4269.tif"
canopy = '/Users/catcoll/Documents/spatial_data/clips/forest_clip_no_data_4269.tif'
land = '/Users/catcoll/Documents/spatial_data/clips/reclassify_land_use_clip.tif'
# dist = '/Users/catcoll/Documents/spatial_data/clips/distance_to_coast.tif'
dist ='/Users/catcoll/Documents/py/isotopes/full_distance.tif'
# residual = '/Users/catcoll/Documents/py/isotopes/clipped_residual_25.tif'
# residual = '/Users/catcoll/Documents/py/isotopes/residual_05_clip_reproj.tif'
residual = '/Users/catcoll/Documents/py/isotopes/high_pass_h2o_removed.tif'

#clip if necessary
output_raster_path = '/Users/catcoll/Documents/py/isotopes/high_pass_h2o_removed_clip.tif'

#clipping shape file
clip=gpd.read_file("/Users/catcoll/Documents/spatial_data/clips/vector_for_clipping.shp")

# tropo = '/Users/catcoll/Documents/py/isotopes/rma_residual.tif'

crs=clip.crs
geometries = [mapping(geom) for geom in clip.geometry]

with rasterio.open(residual) as src:
    nodata=src.nodata
    # Clip the raster with the shapefile geometries
    out_image, out_transform = mask(src, geometries, crop=True, nodata=nodata)

    # Set the no-data value in the clipped raster
    if nodata is not None:
        out_image = np.where(out_image == nodata, np.nan, out_image) 

    # Update metadata for the output file
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "count": 1,
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "crs":crs,
        "nodata": src.nodata
    })
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)
        
clip_res = output_raster_path


#%% run Functions


#resample rasters
"""
rasters already made, rerun if calculating from a new resolution

"""
# output_raster_path = '/Users/catcoll/Documents/py/isotopes/exclude_h2o_column_clip.tif'
resample_raster(canopy, '/Users/catcoll/Documents/spatial_data/clips/forest_clip_resampled_rma_res.tif', output_raster_path, "bilinear")
resample_raster(imper, "/Users/catcoll/Documents/spatial_data/clips/imper_clip_resampled_rma_res.tif", output_raster_path, "bilinear")
resample_raster(output_raster_path, "/Users/catcoll/Documents/spatial_data/clips/dist_resampled.tif", residual, "bilinear")
resample_raster(land, '/Users/catcoll/Documents/spatial_data/clips/reclassify_land_use_resampled_rma_res.tif', output_raster_path, "nearest") #use nearest method for categorical raster
# resample_raster(tropo, '/Users/catcoll/Documents/spatial_data/clips/tropo_resampled_h2ocol.tif', output_raster_path, "bilinear") #use nearest method for categorical raster




#%% create data frame


#files for 5km
# numerical_rasters = [
#     '/Users/catcoll/Documents/spatial_data/clips/forest_clip_resampled_05.tif',
#     "/Users/catcoll/Documents/spatial_data/clips/imper_clip_resampled_05.tif"]

# categorical_raster_path = '/Users/catcoll/Documents/spatial_data/clips/reclassify_land_use_resampled_05.tif'
# residual_raster_path = residual


#files for 25km resolution
numerical_rasters = [
    '/Users/catcoll/Documents/spatial_data/clips/forest_clip_resampled_rma_res.tif',
    "/Users/catcoll/Documents/spatial_data/clips/imper_clip_resampled_rma_res.tif",
    '/Users/catcoll/Documents/py/isotopes/test.tif']

categorical_raster_path = '/Users/catcoll/Documents/spatial_data/clips/reclassify_land_use_resampled_rma_res.tif'
residual_raster_path = '/Users/catcoll/Documents/py/isotopes/high_pass_h2o_removed_clip.tif'



#get list of unique files
with rasterio.open(categorical_raster_path) as src:
    data = src.read(1)
    #find all unique classes
    unique_classes = np.unique(data)


df = prepare_data(numerical_rasters, categorical_raster_path, residual_raster_path)
df

#%% remove nan values (based on nodata value in the raster)

#cleaning for 25 km: (comment out when using 5km)
df_cleaned=df.replace(101, np.nan)


# #cleaning for 5km:(comment out when using 25km)
# df_cleaned = df
# df_cleaned['residual']=df['residual'].replace(0, np.nan)

#cleaning for both
df_cleaned = df_cleaned.dropna()
df_cleaned=df_cleaned.drop(['one_hot_0', 'one_hot_255'], axis=1)

#save file

output_file = '/Users/catcoll/Documents/spatial_data/high_pass_h2o_removed.xlsx'

df_cleaned.to_excel(output_file, index=False)  # index=False to avoid saving row indices
#%% Train Random Forest Model

file ='/Users/catcoll/Documents/spatial_data/high_pass_h2o_removed.xlsx'
data = pd.read_excel(file)
data.columns

#change distance to coast feature to categorical (all values between 0-1 are True)
#comment out when using continuous distance to coast feature
coast = data[['numerical_feature_2']]
c=coast.values
binary_coast=(c<1).astype(int)
new_data=data.drop(['numerical_feature_2'], axis = 1)
new_data.columns
new_data.insert(2, 'numerical_feature_2', binary_coast)
data=new_data


#feature columns
X = data[['numerical_feature_0', 'numerical_feature_1', 'numerical_feature_2','one_hot_11',
       'one_hot_21', 'one_hot_31', 'one_hot_41', 'one_hot_51', 'one_hot_71',
       'one_hot_81', 'one_hot_90']]



#target column
y = data['residual']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#%% model evaluation
from sklearn.metrics import mean_squared_error, r2_score
# Making predictions on the same data or new data
predictions = model.predict(X_test)
 
# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

y_pred = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
r2 = r2_score(y_train, y_pred)

#%% Feature Importance
#get and plot feature importance
importances = model.feature_importances_

labels = ["canopy", "imper", 'dist to coast','open_water', 'urban', 
          'barren', 'forest', 'scrub', 'moss/grass',
          'agriculture','wetlands']

feature_importances_data = pd.DataFrame({
    'Feature': labels,
    'Importance': importances
    })

feature_importances_data = feature_importances_data.sort_values(by='Importance', ascending=False)

print(feature_importances_data)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_data['Feature'], feature_importances_data['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest Regressor (high pass h2o removed, binary coast)')
plt.gca().invert_yaxis()
plt.show()

#%% Partial Dependence Plots

#split numerical and categorical features
numerical_features = ['numerical_feature_0', 'numerical_feature_1']
categorical_features = X_train.columns.drop(numerical_features)



features_info = {
    # features of interest
    "features": ['numerical_feature_0', 'numerical_feature_1','one_hot_51','one_hot_90'],
    # type of partial dependence plot
    "kind": "average",
    # information regarding categorical features
    "categorical_features": categorical_features,
}
_, ax = plt.subplots(ncols=2, nrows=2, figsize=(9, 8), constrained_layout=True)
display = PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    **features_info,
    ax=ax,
    
)
display.figure_.suptitle(
    (
        "Partial dependence for RF Regressor"
        "\n (highest ranked features)"
    ),
    fontsize=16,
)


#%%Extras

#plotting no data
# x = '/Users/catcoll/Documents/spatial_data/clips/forest_clip_resampled.tif'
# from rasterio.plot import show
# import matplotlib.pyplot as plt

# # Open the raster file
# res = rasterio.open(x)

# # Create a figure and axis
# fig, ax = plt.subplots()

# # Display the raster data
# show(res, ax=ax, cmap='viridis')

# cbar = plt.colorbar(ax.images[0], ax=ax)
# cbar.set_label('Residual Values')  # Label for the colorbar

# res.nodata


