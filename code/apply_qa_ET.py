import sys 
sys.path.append('/workspace/TATSSI/')
import collections
import json
import glob

from TATSSI.notebooks.helpers.qa_analytics import Analytics
# from TATSSI.notebooks.helpers.utils import *
from TATSSI.input_output.utils import *
from TATSSI.notebooks.helpers.time_series_interpolation import \
                TimeSeriesInterpolation


sources = [
#        '/wp_data/sites/Degero/MODIS/MOD16A2GF.061/h18v02',
 #       '/wp_data/sites/kampar_1/MODIS/MOD16A2GF.061/h28v08',
  #      '/wp_data/sites/kampar_2/MODIS/MOD16A2GF.061/h28v08',
   #     '/wp_data/sites/kampar_3/MODIS/MOD16A2GF.061/h28v08',
    #    '/wp_data/sites/CongoNorth/MODIS/MOD16A2GF.061/h19v08',
     #   '/wp_data/sites/CongoSouth/MODIS/MOD16A2GF.061/h19v09',
        '/wp_data/sites/Norfolk/MODIS/MOD16A2GF.061/h18v03',
      #  '/wp_data/sites/MoorHouse/MODIS/MOD16A2GF.061/h17v03',
       # '/wp_data/sites/HatfieldThorne/MODIS/MOD16A2GF.061/h17v03',
        #'/wp_data/sites/Gnarrenburger/MODIS/MOD16A2GF.061/h18v03',
        #'/wp_data/sites/MerBleue/MODIS/MOD16A2GF.061/h12v04']
        ]
outputs = [   
        #'/wp_data/sites/Degero/MODIS/analytics/',
       # '/wp_data/sites/kampar_1/MODIS/analytics/',
       # '/wp_data/sites/kampar_2/MODIS/analytics/',
       # '/wp_data/sites/kampar_3/MODIS/analytics/', 
       # '/wp_data/sites/CongoNorth/MODIS/analytics/', 
       # '/wp_data/sites/CongoSouth/MODIS/analytics/',
        '/wp_data/sites/Norfolk/MODIS/analytics/',
       # '/wp_data/sites/MoorHouse/MODIS/analytics/', 
       # '/wp_data/sites/HatfieldThorne/MODIS/analytics/',
       # '/wp_data/sites/Gnarrenburger/MODIS/analytics/',
       # '/wp_data/sites/MerBleue/MODIS/analytics/']
       ]
d = dict(zip(sources, outputs))

for source_dir, output_dir in d.items(): 

    print(source_dir)
    product = 'MOD16A2GF'
    version = '061'
    start_date = '01-06-2013' # dd-mm-yyyy
    end_date = '30-06-2023'
    qa_def = 'ET_QC_500m'
    fname = "/workspace/WorldPeatland/QA_settings/MOD16A2GF.061_ET_QC_500m.json"
    # set the data variable to be masked 
    _data_var = '_ET_500m'

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
    with open(fname, 'r') as f:
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

    # Add one dimension and save to disk percentage of data avail.
    tmp_data_array = qa_analytics.pct_data_available.expand_dims(
            dim='time', axis=0)

    save_dask_array(fname=f'{output_dir}/{product}.{version}_pct_data_available.tif',
            data=tmp_data_array,
            data_var=None, method=None)

    # Add one dimension and save to disk max gap-length
    tmp_data_array = qa_analytics.max_gap_length.expand_dims(
            dim='time', axis=0)

    save_dask_array(fname=f'{output_dir}/{product}.{version}_max_gap_length.tif',
            data=tmp_data_array,
            data_var=None, method=None)

    # Save mask
    save_dask_array(fname=f'{output_dir}/{product}.{version}_qa_analytics_mask.tif',
            data=qa_analytics.mask,
            data_var=None, method=None)

    # Interpolate
    qa_analytics.selected_data_var = _data_var
    qa_analytics.selected_interpolation_method = 'linear'

    tsi = TimeSeriesInterpolation(qa_analytics, isNotebook=False)
    tsi.interpolate(progressBar=None)

