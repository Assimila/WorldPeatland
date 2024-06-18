from osgeo import gdal, osr
import numpy as np
import xarray as xr
import glob
import yaml
import logging
import os
from pathlib import Path
from datetime import datetime as dt

import sys
sys.path.insert(0,'/workspace/WorldPeatland/code/')
from save_xarray_to_gtiff_old import *

from gdal_sheep import *
from MLEO_NN import *
from smoothn import smoothn

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def create_dir(output_dir, directory):
    
    '''
    create_dir fucntion will first check if the directory already exist if not it will 
    create a directory where it will store the data to be downloaded
    
    INPUTS:
        - output_dir (str/path) - specified by the user where they want the data to be downloaded
        - directory (str) - specified by each step in the code to create, usually its the name of the data product to be downloaded
    '''

    # Path 
    path = os.path.join(output_dir, directory) 
    
    if not os.path.exists(path):
        os.makedirs(path)
        LOG.info(f"Directory '{path}' created successfully.")
    else:
        LOG.info(f"Directory '{path}' already exists.")

    return path

def get_file_name(file_path):
    
    '''get_file_name from the file path returns the modis data product name
    and the version 
    
    INPUT
        - file_path (str) - it would be the one set by the user when running the downloader_wp
            + MODIS the path specific to download MODIS data
    OUTPUT
        - file_name[0] (str) - in this case it would be the MODIS data product name
        - file_name[1] (str) - in this case it would be the MODIS data product version
    '''
    
    file_path_components = file_path.split('/')
    file_name = file_path_components[-1].rsplit('.', 1)
    return file_name[0], file_name[1]

def create_xr(orbit, output_dir, site_name):
    
    '''
    create_xr function takes as input asc or desc and returns an xr with 3 data variables VV, VH and angles
    of the input orbit. 
    
    INPUTS:
        - orbit (string) - ASCENDING or DESCENDING

    OUTPUTS:
        - ds (xarray.Dataset) - 

    '''
    
    # get the required band_name and creates a list of products needed to extract corresponding files 
    bd = [f'VV_{orbit}',f'VH_{orbit}',f'angle_{orbit}']
    
    # call the get_list_of_files function to get the nested list each key is a band name
    # and the values are the corresponding file paths for the different days the image was captured for this band 
    nested = [sorted(glob.glob(f'{site_directory}/Sentinel/datacube/S1_GRD/%s/{tile_name}/*.tif'%i)) for i in bd]
    # create a dict to keep track of the name of the band for each list of file paths
    fdict = dict(zip(bd, nested))
    
    stack_list = []
    for i in fdict:
        stack, dts, opn = gdal_stack_dt(fdict[i], 'time') # time attribute should be 'time' for all s1 images
        stack_list.append(stack)
    stack_dict = dict(zip(bd, stack_list))
    
    xs, ys = create_coord_list(opn)
    variable_name = bd
    ds = xr.Dataset(data_vars = {i:(('time','latitude', 'longitude'),stack_dict[i])for i in bd},
                   coords={'time': dts,
                          'latitude': ys,
                          'longitude': xs})
    
    return ds

def apply_threshold(ds):
    
    for var_name in ds.data_vars:
        if var_name.startswith('V'):
            ds[var_name] = ds[var_name].where(ds[var_name] > -30, np.nan)
    return ds

def calc_cr(ds, orbit):
    
    '''
    calc_cr function takes as input the xarray.dataset, returns a new xarray with cross ratio calculated,  
    and saves it as a netcdf file. 
    
    INPUTS:
        - orbit (string) - ASCENDING or DESCENDING

    OUTPUTS:
        - ds (xarray) - with cross ratio as the 4th variable (in total 4 variables)

    '''
    
    # Calculate Cross Ratio 
    ds = ds.assign(cr = ds[f"VH_{orbit}"] - ds[f'VV_{orbit}'])
    # Save as netcdf
    ds.to_netcdf(f'{site_directory}/Sentinel/cross_ratio_{orbit}.nc')
    
    return ds

def transform_save(ds, orbit, saved_path):
    
    # save_xarray can only save one variable
    # do we create a new one that can fit more than one variable?
    # because need to save observation and cross ratio 
    # ==>> maybe try to to save xarray with many variables for each orbit. 
    'ex: ascending orbit has an xarray with cross ratio, VV and VH'
    output_utm = saved_path + f'/cross_ratio_{orbit}_utm.tif'
#     save_xarray_old(output_utm, ds, 'cr')
    
    # change projection from utm to sinusoidal 
    output_sinu = saved_path + f'/cross_ratio_{orbit}_sinusoidal_resampled.tif'
    proj4_string = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs '
    ds = gdal.Open(output_utm)

    # reproject to sinusoidal and resample to 10 by 10 pixel size
    dsReprj = gdal.Warp(output_sinu, ds, dstSRS=proj4_string, xRes = 10, yRes = 10)
    ds = dsReprj = None # close the files
    
    # delete utm files
    os.remove(output_utm)
    
    return output_sinu

def main(site_directory, output_dir):
    
    # get the site name from site_directory
    path_components = site_directory.split(os.sep)
    site_name = path_components[-1]


    orbits = ['ASCENDING', 'DESCENDING']
    
    for orbit in orbits:

        ds = create_xr(orbit, output_dir , site_name)

        _ds = apply_threshold(ds)

        # TODO seperate per angles??

        ds_cr = calc_cr(_ds, orbit)

        # set the current projection taken from linux gdalinfo -proj4 from a random sentinel-1 tif 
        # TODO automatic getting the projection for each zone 
        # think about if 2 zones for one site 
        # then will have to get the zone that is covering the largest area
        ds_cr.attrs['crs'] = '+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs '

        saved_path = create_dir(f'{site_directory}/Sentinel/', 'CrossRatio')
        
        output_utm = saved_path + f'/cross_ratio_{orbit}_utm.tif'
        save_xarray_old(output_utm, ds, 'cr')

        # change projection from utm to sinusoidal
#         output_sinu = transform_save(ds_cr, orbit, saved_path)
#         LOG.info(f'Cross Ratio for {orbit} has been saved here {saved_path}')

#         # Create an xarray to be able to perform smoothn 
#         arr, dts, saved_opn = gdal_dt(output_sinu, 'time')
#         ds = create_xarr(saved_opn, 'cr', arr, dts)

        # smoothn return first the smoothend data and some other quality data...
        s_smoothn = smoothn(y=ds_cr['cr'], isrobust=True, axis=0)[0]

        # Save pixel level data final output after processing into geotif
        ds_s = create_xarr(saved_opn, 'cr_smooth', s_smoothn, dts)

#         # add the crs as attribute 
#         ds_s.attrs['crs'] = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs '

        fname = saved_path + f'/cross_ratio_{orbit}_utm_resampled_smoothn.tif'
        save_xarray_old(fname, ds_s, f'cr_smooth')
        LOG.info(f'Cross Ratio Smoothened for {orbit} has been saved here fname')

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <site_directory> <output_dir>") # the user has to input two arguments  
    else:
        site_directory = sys.argv[1] 
        output_dir = sys.argv[2] 
        main(site_directory, output_dir)
        
        
        
        
