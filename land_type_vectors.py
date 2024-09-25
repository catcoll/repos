#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:46:41 2024

@author: catcoll

In this code:
    
    imports land use categorical raster file
    Creates mask for given unique value in categorical raster
    creates individual shapefiles for each land use type and saves to a directory
    
"""

import os
import rasterio
import numpy as np
import shapely.geometry
import geopandas as gpd
import rasterio.features

raster_file = '/Users/catcoll/Documents/spatial_data/clips/reclassify_land_use_resampled_05.tif'

def raster_to_shapefile(raster_file, values, output_dir):
    with rasterio.open(raster_file) as src:
        image = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for value in values:
        # Create a mask where the value matches
        mask = (image == value).astype(np.uint8)
    
        # Extract shapes from the mask
        shapes = rasterio.features.shapes(mask, transform=transform)
    
        # Create polygons from the shapes
        polygons = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]
    
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    
        # Define output shapefile path
        output_vector_file = os.path.join(output_dir, f'land_use_{value}_05.shp')
    
        # Save to shapefile
        gdf.to_file(output_vector_file)

# Example usage
raster_file
values = [11,21,31,41,51,71,81,90]
output_dir = '/Users/catcoll/Documents/spatial_data/land_type_vectors/vectors_05'
raster_to_shapefile(raster_file, values, output_dir)
