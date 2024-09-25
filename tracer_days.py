#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:34:15 2024

@author: catcoll

Loop through tracer days (or specified date range) and collect file names

"""


import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_tracer_days(start_dir, months):
    ''''
    Input: starting directory (year), file path
        months: months of interest, str
        
    output: list of file paths
    '''
    
    file_list=[]
    
    for root, dirs, files in os.walk(start_dir):
        
        month_dir= os.path.basename(root)
        if month_dir in months:
            for day_dir in dirs:
                day_dir_path = os.path.join(root, day_dir)
                
                for filename in os.listdir(day_dir_path):
                    file_path = os.path.join(day_dir_path, filename)

                
                    try:
                        coords = xr.open_dataset(file_path, group="instrument")
                        lon = coords['longitude_center'].values
                        lat = coords['latitude_center'].values
                        
                        xy=np.column_stack((lon,lat))
                                
                                # Iterate over each pair of x, y values
                        for i in range(xy.shape[0]):
                            x = xy[i, 0]
                            y = xy[i, 1]
                                    
                                    # Example condition: x between -100 and -50, y between 30 and 60
                            if -96 <= x <= -93 and 28 <= y <= 30:
                                file_list.append(file_path)
                                break 
                        coords.close()

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    return file_list

#%%

#time frame of interest in 2021
days_in_2021 = "/Volumes/waterisotopes/HDO_trim_CLEAR/2021"
months_2021 = ["10", "11", "12"]

houston_2021 = get_tracer_days(days_in_2021, months_2021)

houston_2021
len(houston_2021)

#time frame of interest in 2022
days_in_2022 = "/Volumes/waterisotopes/HDO_trim_CLEAR/2022"
months_2022 = ["01","02","03","04","05","06","07","08","09"]

houston_2022 = get_tracer_days(days_in_2022,months_2022)
len(houston_2022)
type(houston_2022)

#%% save file names to txt file
# open file in write mode
with open(r'/Users/catcoll/Documents/py/isotopes/tracer_files', 'w') as fp:
    for item in houston_2021:
        # write each item on a new line
        fp.write("%s\n" % item)
    for i in houston_2022:
        fp.write("%s\n" % i)

