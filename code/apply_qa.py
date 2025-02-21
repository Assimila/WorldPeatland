
import json
import collections
import glob
import shutil

import numpy as np
import rioxarray
from datetime import datetime
import os
from osgeo import gdal
import logging
from WorldPeatland.code.download_modis import read_config
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from TATSSI.TATSSI.notebooks.helpers.qa_analytics import Analytics
from TATSSI.TATSSI.input_output.utils import save_dask_array
from TATSSI.TATSSI.notebooks.helpers.time_series_interpolation import \
    TimeSeriesInterpolation
from WorldPeatland.code.utils import create_dir

import sys
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


''' 
apply_qa its the 3rd code to run will apply the qa settings to the MODIS data time series generated in code 2 
then it will INTERPOLATE the cleaned time series
'''


def main(site_directory, qa_path):

    # get the specific config path for this site 
    # to get the dates and the products

    # get the site name from site_fpath
    config = glob.glob(site_directory + f'*_config.yml')
    config_fname = config[0]

    start_date, end_date, products = read_config(config_fname)

    # change date format 
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d-%m-%Y')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d-%m-%Y')

    for i in range(len(products)):

        product, version = products[i]['product'].split('.')

        LOG.info(f' product processed is {product} version {version}')

        # Skip processing if product is MCD64A1 or MCD43A2
        if product in ('MCD64A1', 'MCD43A2'):
            LOG.info(f'Skipping product {product}')
            continue

        # if MCD43A3 apply snow mask before interpolation
        if product == 'MCD43A3':

            albedo_sr_paths = glob.glob(os.path.join(site_directory, 'MODIS/MCD43A3.061/*/Albedo_WSA_Band2/*.tif'))
            for albedo_path in albedo_sr_paths:
                # get the timestep
                timestep = os.path.basename(albedo_path).split('.')[1]

                # find the corresponding snow albedo tif
                snow_path = glob.glob(os.path.join(site_directory,
                                                   f'MODIS/MCD43A2.061/*/Snow_BRDF_Albedo/MCD43A2.{timestep}.*.tif'))

                if snow_path is not None:
                    snow_path = snow_path[0]
                else:
                    LOG.error(f'No corresponding snow BRDF albedo was found for {timestep}')

                dataset = gdal.Open(albedo_path, gdal.GA_Update)
                band = dataset.GetRasterBand(1)
                albedo_arr = band.ReadAsArray()

                dataset = gdal.Open(snow_path)
                bd = dataset.GetRasterBand(1)
                snow_arr = bd.ReadAsArray()

                # apply snow mask
                # Treat snow == 255 (fill value) as 0 (no snow)
                snow_arr = np.where(snow_arr == 255, 0, snow_arr)
                # Apply the mask: keep albedo where snow == 1, else set to 32767 (fil_value)
                masked_albedo = np.where(snow_arr != 1, albedo_arr, 32767)

                band.WriteArray(masked_albedo)
                band.FlushCache()
                bd.FlushCache()
                dataset = None

                LOG.info(f'snow masked applied to {albedo_path}')

        _data_var_list = products[i]['data_var']
        qa_def_list = products[i]['qa_def']

        # zip the list to match to the data_var to the corresponding qa_def
        for _data_var, qa_def in zip(_data_var_list, qa_def_list):

            # source_dir where the modis data for this product is stored
            source_dirs = glob.glob(site_directory + 'MODIS/' + f'{product}.{version}/*/')

            for source_dir in source_dirs:

                # check if file exists in this directory
                if not os.path.exists(source_dir):
                    LOG.error(f'A MODIS file does not exist: {source_dir}')

                # json file for qa_settings
                qa_json = os.path.join(
                    qa_path, f"{product}.{version}_{qa_def}.json"
                )

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
                    data_format='tif'
                )

                # Get QA definition
                for idx, _def in enumerate(qa_analytics.qa_defs):
                    layer = _def['QualityLayer'].unique()[0]
                    if layer == qa_def:
                        index = idx

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
                QA_settings = os.path.basename(qa_path)
                path_analytics = create_dir(site_directory + 'MODIS/', f'analytics_{QA_settings}')

                print('path_analytics:', path_analytics, 'for variable:', _data_var)

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

                # TODO write for users when and how to change the min_obs_ratio
                min_obs_ratio = 0

                tsi = TimeSeriesInterpolation(qa_analytics, min_obs_ratio=min_obs_ratio, isNotebook=False)
                tsi.interpolate(progressBar=None)

                LOG.info(f'Data {_data_var} has been interpolated')


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <site_root_data_dir> <qa_path>")  # the user has to input one argument
    else:
        site_directory = sys.argv[1]
        qa_path = sys.argv[2]
        main(site_directory, qa_path)
# Choose the QA settings folder that you'd like the strict one is 'QA_settings'
# while the least restrictive options is 'QA_settings_v1'

# # example of user input arguments
# python apply_qa.py /data/sites/Norfolk/
