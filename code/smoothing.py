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

from TATSSI.time_series.smoothing import Smoothing

import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

'''smoothing.py is the 4th code to run it will apply the smooth factor from the config file to the cleaned and interpolated time series'''


# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s %(message)s')
# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
LOG.addHandler(ch)

def main(directory, site_name):

    # get the specific config path for this site 
    # to get the dates and the products
    config_dir = directory + site_name + f'/{site_name}_config.yml'
    start_date, end_date, products = read_config(config_dir)
    products

    for i , j  in enumerate(products):

        product = j['product']
        
        if product == 'MCD64A1.061':
            continue
        
        smoothing_method, s = j['smooth_method'], j['smooth_factor']

        pattern = directory + f'{site_name}/MODIS/{product}/*/interpolated/*linear.tif'
        f_list = glob.glob(pattern)

        for fname in f_list:
            
            # Split the file name into base and extension
            base, extension = fname.rsplit('.', 1)
            output_fname = f"{base}.{smoothing_method}.{s}.{extension}"
            
            LOG.info(f'smoothing started for {fname} with {smoothing_method} as smoothing method and this factor {s}')
              
            smoother = Smoothing(data=None, fname=fname,
                                 output_fname=output_fname,
                                 smoothing_method=smoothing_method,
                                 s=float(s), progressBar=None)

            smoother.smooth()
            
            LOG.info(f'saved in this directory {output_fname}')
            
            
if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <directory> <site_name>") # the user has to input two arguments  
    else:
        directory = sys.argv[1]
        site_name = sys.argv[2] 
        main(directory, site_name)
        
# # example of user input arguments
# python apply_qa.py /data/world_peatlands/demo/dry_run/ Norfolk