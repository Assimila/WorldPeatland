import os
import subprocess
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime

import sys
sys.path.append('/home/ysarrouh/TATSSI/')
sys.path.insert(0,'/home/ysarrouh/WorldPeatlands/')
from downloader_wp_test import *
from gdal_sheep import *
from save_xarray_to_gtiff_old import *


from TATSSI.time_series.smoothing import Smoothing

import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


'''piexl_ts is the 5th code to run it will apply the scaling factor and detrend the time series'''


def pixel_ts(path, _data_var, scaling_factor, period, site_name, detrend):
    
    '''
    pixel_ts function will open the tatssi geotif linear and smoothened files, it will multiply it
    by the corresponding scaling factor and then detrend it. detrend is set as true, if you wish 
    to have the pixel data not detrended for zonal statistics or other processing purposes set detrend 
    as false
    
    INPUTS:
        - data_tif_path (string) - path to the tatssi output tif file, usually linear and smoothened
        - nm_lyr (string) - also name of the layer in the tif file 
    
    OUTPUTS:
        - fname (String) - saved tif file path 

    '''
    
    # open the tif and extract dataframe
    arr, dts, saved_opn = gdal_dt(path, 'RANGEBEGINNINGDATE')
    # put the tiff in an xarray
    ds = create_xarr(saved_opn, _data_var, arr, dts)
    
    # multiply by scaling factor
    ds = ds*float(scaling_factor)
    LOG.info(f'multiplying by the scale factor of {scaling_factor}')

    base, extension = os.path.splitext(os.path.basename(path))

    path_analytics = create_dir(directory + site_name + '/MODIS/', 'timeSeries')
    
    # period dictionary in days according to the data_var
    # period = 365/temporal resolution of the layer 
    
    if detrend:
        
        new_fname = f"{base}.descaled.detrended{extension}"
        output_fname = f"{path_analytics}/{new_fname}"

        # detrend ts 
        trend_ds = ds[_data_var].rolling(time = int(period), min_periods=1,center=True).mean()
        ds_output = trend_ds.to_dataset(name=_data_var)
      
    # if ds was not detrended...
    else: 
        
        new_fname = f"{base}.descaled{extension}"
        output_fname = f"{path_analytics}/{new_fname}"
        
        ds_output = ds
    
    # add the crs to the attributes of the xarray so it is there when saving the detrended final ts tif
    # in this case we are making all projection sinusoidal like tatssi
    ds_output.attrs['crs'] = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs '

    # save the xarray into a tif file 
    save_xarray_old(output_fname, ds_output, _data_var)
    # return the path of the new saved tif to perform zonal statistics on tif already multiplied by 
    # scaling factor and detrended 
    
    LOG.info(f'file saved here {output_fname}')
    return


def main(directory, site_name, value):

    # get the specific config path for this site 
    # to get the dates and the products
    config_dir = directory + site_name + f'/{site_name}_config.yml'
    start_date, end_date, products = read_config(config_dir)
    
    
    for i, j in enumerate(products):
    
        product = j['product']

        if product == 'MCD64A1.061':
            continue
    
        _data_var_list = j['data_var'] 

        for _data_var in _data_var_list:

            scaling_factor, s, smoothing_method, period = j['scaling_factor'], j['smooth_factor'], j['smooth_method'], j['period']

            pattern = directory + f'{site_name}/MODIS/{product}/*/interpolated/*.{_data_var}.linear.{smoothing_method}.{s}.tif'
            path = glob.glob(pattern)[0] 
            
            LOG.info(f'Processing this file {path}')

            pixel_ts(path, _data_var, scaling_factor, period, site_name, detrend = value)

    
if __name__ == "__main__":

    if len(sys.argv) != 4:

        print("Usage: python script.py <directory> <site_name>") # the user has to input two arguments  
    else:
        directory = sys.argv[1]
        site_name = sys.argv[2] 
        value = sys.argv[3].lower()
        
        if value == 'true':
            value = True
        elif value == 'false':
            value = False
        else:
            print("Invalid value! Please enter 'True' or 'False'.")
            sys.exit(1)
        
        main(directory, site_name, value)

# example of user input arguments
# python pixel_ts.py /data/world_peatlands/demo/dry_run/ Norfolk True (True or False) 
    
