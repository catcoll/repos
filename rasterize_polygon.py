#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:40:46 2024

@author: catcoll

Make a raster out of a polygon
"""

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import shape
import numpy as np
from scipy.ndimage import distance_transform_edt


#where to save raster
output_raster_path = '/Users/catcoll/Documents/spatial_data/clips/ocean_raster.tif'


#%% rasterize polygon with reference raster 

# Load the grid raster to get its properties to use as reference
#can be any raster with target resolution and projection
with rasterio.open('/Users/catcoll/Documents/spatial_data/clips/reclassify_land_use_resampled_exclude_h2o.tif') as src:
    # Get the resolution
    pixel_width, pixel_height = src.res
    # Get the transform and CRS
    transform = src.transform
    crs = src.crs



# Load your polygon (ocean) and grid raster
ocean_polygon = gpd.read_file('/Users/catcoll/Documents/spatial_data/clips/coast.shp')

bounds = ocean_polygon.total_bounds
left, bottom, right, top = bounds

width = int((right - left) / pixel_width)
height = int((top - bottom) / pixel_height)


dtype = np.uint8
nodata = 0

#rasterize
with rasterio.open(
    output_raster_path, 'w', driver='GTiff',
    height=height, width=width,
    count=1, dtype=dtype,
    crs=crs,  # Use the CRS from the reference raster
    transform=transform,
    nodata=nodata
) as dst:
    dst.write(np.zeros((height, width), dtype=dtype), 1)

# Rasterize the polygon
with rasterio.open(output_raster_path, 'r+') as dst:
    out_shape = (height, width)
    mask = geometry_mask(
        [geom for geom in ocean_polygon.geometry],
        transform=transform,
        invert=True,
        out_shape=out_shape
    )
    
    dst.write(np.where(mask, 1, nodata), 1)
    
    



















