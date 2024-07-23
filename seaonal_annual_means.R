# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import scipy

xr.set_options(keep_attrs=True, display_expand_data=False)
np.set_printoptions(threshold=10, edgeitems=2)

%xmode minimal
%matplotlib inline
%config InlineBackend.figure_format='retina'

#%%



#Upload data
ds=xr.open_dataset("/Users/catcoll/Documents/py/isotopes/TROPOMI_HDO_CLEAR_1x1.nc")
ds

#dd is: delta expression of HDO to H2O ratio
dd=ds.deltad


#Water mass weighted delta expression of HDO to H2O ratio
dw=ds.deltad_weighted

dw

#%%

#plot first time step

lat=dd.lat.data
lon=dd.lon.data
temp=dd.data
plt.figure()
plt.pcolormesh(lon,lat,temp[0,:,:])
temp.mean(axis=1)

dd.isel(time=1).plot(x="lon")
#%%
#average weighted delta fraction over years and seasons


seasonal_mean=ds.groupby("time.season").mean()
seasonal_mean

seasonal_mean = seasonal_mean.sel(season=["DJF", "MAM", "JJA", "SON"])
seasonal_mean

seasonal_mean.deltad_weighted.plot(col="season", col_wrap=2)

annual_mean=ds.groupby("time.year").mean()
annual_mean

annual_mean.deltad_weighted.plot(col="year", col_wrap=3)

#%%
#index for lat lon of the gulf coast region

# "nearest indexing at multiple points"
gulf=ds.sel(lon=[79, 98], lat=[18, 30], method="nearest")

gulf_annual=gulf.groupby("time.year").mean()
gulf_annual.deltad_weighted.plot(col="year", col_wrap=3)


gulf_seasonal=gulf.groupby("time.season").mean()
gulf_seasonal
gulf_seasonal.deltad_weighted.plot(col="season", col_wrap=2)












