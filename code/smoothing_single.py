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
sys.path.append('/workspace/TATSSI/')
sys.path.insert(0,'/workspace/WorldPeatland/code/')
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

def main():

    dirs = glob.glob(f'/wp_data/sites/*/')
    dirs.sort()

    for site_directory in dirs:

        # get the specific config path for this site 
        # to get the dates and the products
    
        # get the site name from site_directory
        config = glob.glob(site_directory + f'*_config.yml') 
        config_fname =  config[0]
        start_date, end_date, products = read_config(config_fname)

        product = 'MOD16A2GF.061'
        
        smoothing_method, s = 'smoothn', '0.5'

        pattern = site_directory + f'/MODIS/MOD16A2GF.061/*/*/interpolated/*linear.tif'
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

    if len(sys.argv) != 1:

        print("Usage: python script.py <site_directory>") # the user has to input one argument
    else:
        main()
        
# # example of user input arguments
# python smoothing.py /data/sites/Norfolk
