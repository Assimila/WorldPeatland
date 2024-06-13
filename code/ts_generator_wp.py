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
sys.path.append('/workspace/TATSSI/')

sys.path.insert(0,'/workspace/WorldPeatland/code/')
from downloader_wp_test import *

from TATSSI.time_series.generator import Generator  
    

'''ts_generator_wp 2nd script to run, it will generate time series for the MODIS data'''    

# need to update TATSSI/TATSSI/qa/EOS catalogue from APPEARS to contain all the updated products and version 
# we can do that by removing the current pkl files in EOS and running TATSSI UI in the download data tab 
# this will initiate by itself getting all the available pkl files in APPEARS

    
# TODO check for existing MODIS time series files 
# it seems to work only in the UI and not in the scripts
    
def get_ts(site_directory):
    
    '''
    INPUT site_directory from user, path to the folder where downloader_wp was run to download all 
    products 
    '''
    modis_dir = site_directory + 'MODIS/'
    print('modis directory', modis_dir) 


    if not os.path.exists(site_directory):
        LOG.error(f'This director does not exist: {site_directory}')
        return
    
    # check if a MODIS file exists in this directory 
    if not os.path.exists(modis_dir):
        LOG.error(f'A MODIS file does not exist: {modis_dir}')
        return
    
    # Getting the list of directories 
    dir_ = os.listdir(modis_dir) 

    # Checking if the list is empty or not 
    if len(dir_) == 0: 
        print(f"This directory {modis_dir} is empty") 
        return 

    for i, n in enumerate(dir_):


        LOG.info(n)


        product_name, version = dir_[i].rsplit('.', 1) 
        
        # since we want to do the processing per tile 
        # glob will get all the tile files for the product we are looping in 
        output_dirs = glob.glob(modis_dir + n +f'/*/')
        
        for output_dir in output_dirs:


            # Create for MCD64A1 because the files are in hdf format just 
            # for the processing to be able to use the data do not apply mask 
            # later on MCD64A1 with the apply_qa.py
        
            if os.path.getsize(output_dir) == 0:
                LOG.error(f'{output_dir} this file is empty')
                continue

            # TATSSI Time Series Generator object
            tsg = Generator(source_dir=output_dir,
                     product=product_name,
                     version=version,
                     data_format='hdf',
                     progressBar=None,
                     preprocessed=True)
            
            LOG.info('++++++++++++++++++++++++++++++++++++++++++')
            LOG.info(output_dir)
            LOG.info(product_name)
            LOG.info(version)




            tsg.generate_time_series()

            LOG.info(f'time series generated for this product {n}')       
        
        
def main(site_directory):
    '''
    INPUT
        site_directory - str - path to a specific site where all data were previously downloaded'''
    get_ts(site_directory)
    
if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python script.py <site_directory>") # the user has to input one argument
    else:
        site_directory = sys.argv[1]
        main(site_directory)


# example how to run this script in command line

# python ts_generator_wp.py /data/sites/Norfolk













    
    
