#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:47:18 2024

@author: catcoll

image filtering:

High pass filter
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import rasterio
import xarray as xr
import rioxarray





ds=xr.open_dataset("/Volumes/waterisotopes/TROPOMI_HDO_CLEAR_p25.nc")

gulf=ds.sel(lat=slice(25,32), lon=slice(258,282))

dw=gulf.deltad_weighted


# Load the raster image using rasterio
with rasterio.open('/Users/catcoll/Documents/py/isotopes/raw_tropo.tif') as src:
    image_array = src.read(1)  # Read the first band

# Perform FFT
f_image = fftpack.fft2(image_array)
f_image_shifted = fftpack.fftshift(f_image)

# Create a Butterworth high-pass filter
def butterworth_highpass(shape, cutoff, order=2):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    radius = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
    filter_mask = 1 / (1 + (cutoff / radius) ** (2 * order))
    filter_mask[center_row, center_col] = 0  # Avoid division by zero
    return 1 - filter_mask  # Invert to create high-pass filter

# Apply the filter
cutoff_frequency = 20 # Adjust as needed
hp_filter = butterworth_highpass(image_array.shape, cutoff_frequency)
filtered_f_image = f_image_shifted * hp_filter

# Inverse FFT
filtered_image = fftpack.ifft2(fftpack.ifftshift(filtered_f_image))
filtered_image_real = np.real(filtered_image)

# Display the result
plt.imshow(filtered_image_real, cmap='viridis')
plt.axis()
plt.colorbar()
plt.show()

annual = gulf.deltad_weighted.mean(dim='time')
annual.coords['lon'] = (annual.coords['lon'] + 180) % 360 - 180
# Set spatial dimensions
annual = annual.rio.set_spatial_dims('lon', 'lat')

aligned_xarr = xr.DataArray(filtered_image_real, dims=annual.dims, coords=annual.coords)
aligned_xarr = aligned_xarr.rio.set_spatial_dims('lon','lat')

#save as raster
aligned_xarr.rio.to_raster(r"test.tif")
