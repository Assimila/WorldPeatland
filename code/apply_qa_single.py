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
    
source_dir = f"/wp_data/sites/Degero/MODIS/MYD11A1.061/h18v02"

output_dir = f"/wp_data/sites/Degero/MODIS/analytics"

product = 'MYD11A1'
version = '061'
start_date = '01-06-2013' # dd-mm-yyyy
end_date = '30-06-2023'

qa_def = 'QC_Day'
fname = "/workspace/WorldPeatland/QA_settings/MYD11A1.061_QC_Day.json"

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
# set the data variable to be masked 
#     _data_var = list(qa_analytics.ts.data.data_vars.keys())[0]
_data_var = '_LST_Day_1km'

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




