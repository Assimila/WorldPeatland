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
import collections

from TATSSI.notebooks.helpers.qa_analytics import Analytics
# from TATSSI.notebooks.helpers.utils import *
from TATSSI.input_output.utils import *
from TATSSI.notebooks.helpers.time_series_interpolation import \
        TimeSeriesInterpolation


''' apply_qa its the 3rd code to run will apply the qa settings to the MODIS data time series generated in code 2 then it will INTERPOLATE the cleaned time series'''


def main(directory, site_name):
    
    qa_path = "/home/ysarrouh/WorldPeatlands/QA_settings/"
    
    # get the specific config path for this site 
    # to get the dates and the products
    config_dir = directory + site_name + f'/{site_name}_config.yml'
    start_date, end_date, products = read_config(config_dir)

    # change date format 
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d-%m-%Y')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d-%m-%Y')

    for i in range(len(products)):
    
        product, version = products[i]['product'].split('.')

        LOG.info(f' product processed is {product} version {version}')

        if product == 'MCD64A1':
            #TODO maybe generate the ts but TATSSI will cut for this type of data 
            continue

        _data_var_list = products[i]['data_var'] 
        qa_def_list = products[i]['qa_def'] 
        
        # zip the list to match to the data_var to the corresponding qa_def
        for _data_var, qa_def in zip(_data_var_list, qa_def_list):

            # source_dir where the modis data for this product is stored
            source_dir = directory + site_name + '/MODIS/'+ f'{product}.{version}'
            
            # check if file exists in this directory 
            if not os.path.exists(source_dir):
                LOG.error(f'A MODIS file does not exist: {source_dir}')

            # json file for qa_settings
            qa_json = f'{qa_path}{product}.{version}_{qa_def}.json'
                
            # check if a qa_file exists in this directory 
            if not os.path.exists(qa_json):
                LOG.error(f'A MODIS file does not exist: {qa_json}')
            
            # Create the QA analytics object
            qa_analytics = Analytics(
                source_dir=source_dir,
                product=product,
                chunked=True,
                version=version,
                start=start_date,
                end=end_date,
                data_format='tif')

            # Get QA definition
            for i,_def in enumerate(qa_analytics.qa_defs):
                layer = _def['QualityLayer'].unique()[0]
                if layer == qa_def:
                    index = i 

            qa_analytics.qa_def = qa_analytics.qa_defs[index]

            # Set the QA user selection from saved settings
            with open(qa_json, 'r') as f:
                tmp_user_qa_selection = collections.OrderedDict(json.loads(f.read()))

            qa_analytics.user_qa_selection = tmp_user_qa_selection

            # Apply QA analytics - no progress bar
            qa_analytics._analytics(b=None)

            # Save mask and analytics
            # Copy metadata
            qa_analytics.pct_data_available.attrs = \
               qa_analytics.ts.data[_data_var].attrs
            qa_analytics.max_gap_length.attrs = \
               qa_analytics.ts.data[_data_var].attrs
            
            # create the directory to store QA analytics
            path_analytics = create_dir(directory + site_name + '/MODIS/', 'analytics')

            # Add one dimension and save to disk percentage of data avail.
            tmp_data_array = qa_analytics.pct_data_available.expand_dims(
                dim='time', axis=0)
            save_dask_array(fname=f'{path_analytics}/{_data_var}_pct_data_available.tif',
                data=tmp_data_array,
                data_var=None, method=None)

            # Add one dimension and save to disk max gap-length
            tmp_data_array = qa_analytics.max_gap_length.expand_dims(
                dim='time', axis=0)
            save_dask_array(fname=f'{path_analytics}/{_data_var}_max_gap_length.tif',
                data=tmp_data_array,
                data_var=None, method=None)

            # Save mask
            save_dask_array(fname=f'{path_analytics}/{_data_var}_qa_analytics_mask.tif',
                data=qa_analytics.mask,
                data_var=None, method=None)
            
            LOG.info(f'interpolation has started')
            
            # Interpolate
            qa_analytics.selected_data_var = _data_var
            qa_analytics.selected_interpolation_method = 'linear'

            tsi = TimeSeriesInterpolation(qa_analytics, isNotebook=False)
            tsi.interpolate(progressBar=None)

            LOG.info(f'Data {_data_var} has been interpolated')
    
if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <directory> <site_name>") # the user has to input two arguments  
    else:
        directory = sys.argv[1]
        site_name = sys.argv[2] 
        main(directory, site_name)
        
# # example of user input arguments
# python apply_qa.py /data/world_peatlands/demo/dry_run/ Norfolk
