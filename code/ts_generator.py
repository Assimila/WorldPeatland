import yaml
from datetime import datetime, timedelta
import calendar
from tqdm import *
from osgeo import ogr
import shutil
import glob
import logging
from urllib.request import urlretrieve
import os
import json

import sys
sys.path.append('/home/ysarrouh/TATSSI/')

sys.path.insert(0,'/home/ysarrouh/WorldPeatlands/')
from downloader_wp_test import *

from TATSSI.time_series.generator import Generator  
    

# TODO check for existing MODIS time series files 
# it seems to work only in the UI and not in the scripts
    
def get_ts(directory, site_name):
    
    '''
    INPUT directory from user, path to the folder where downloader_wp was run to download all 
    products 
    '''
    modis_dir = directory + site_name + '/MODIS/'
    
    if not os.path.exists(directory):
        LOG.error(f'This director does not exist: {directory}')
        return
    
    # check if a directory with this site_name exists
    if not os.path.exists(directory + site_name):
        LOG.error(f'A file with this site name {site_name} in this directory {directory}'/ 
                  ' does not exist')
        return
    
    # check if a MODIS file exists in this directory 
    if not os.path.exists(modis_dir):
        LOG.error(f'A MODIS file does not exist: {modis_dir}')
        return
    
    # Getting the list of directories 
    dir = os.listdir(modis_dir) 

    # Checking if the list is empty or not 
    if len(dir) == 0: 
        print(f"This directory {modis_dir} is empty") 
        return 

    for i, n in enumerate(dir):

        product_name, version = dir[i].rsplit('.', 1) 
        
        output_dir = modis_dir + n
        
        if product_name == 'MCD64A1':
            continue # do not create ts for these fire related products
            
        else: 
        
            if os.path.getsize(output_dir) == 0:
                LOG.error(f'{output_dir} this file is empty')
                continue

            # TATSSI Time Series Generator object
            tsg = Generator(source_dir=output_dir, product=product_name,
                        version=version, data_format='hdf',
                        progressBar=None, preprocessed=True)

            tsg.generate_time_series()

            LOG.info(f'time series generated for this product {n}')       
        
        
def main(directory, site_name):
    
    get_ts(directory, site_name)
    
if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <directory> <site_name>") # the user has to input two arguments  
    else:
        directory = sys.argv[1]
        site_name = sys.argv[2] 
        main(directory, site_name)
















    
    