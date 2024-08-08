
from glob import glob

from WorldPeatland.code.downloader_wp_test import *
from WorldPeatland.code.gdal_sheep import *
from WorldPeatland.code.save_xarray_to_gtiff_old import *

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


'''piexl_ts is the 5th code to run it will apply the scaling factor and detrend the time series'''


def pixel_ts(path, site_directory, _data_var, scaling_factor, period, detrend):
    
    """
    pixel_ts function will open the tatssi geotiff linear and smoothened files, it will multiply it
    by the corresponding scaling factor and then detrend it. detrend is set as true, if you wish 
    to have the pixel data not detrended for zonal statistics or other processing purposes set detrend 
    as false
    
    INPUTS:
        - data_tif_path (string) - path to the tatssi output tif file, usually linear and smoothened
        - nm_lyr (string) - also name of the layer in the tif file 
    
    OUTPUTS:
        - fname (String) - saved tif file path 

    """
    
    # open the tif and extract dataframe
    arr, dts, saved_opn = gdal_dt(path, 'RANGEBEGINNINGDATE')
    # put the tiff in an xarray
    ds = create_xarr(saved_opn, _data_var, arr, dts)
    
    if _data_var == '_Lai_500m':
        scaling_factor = '0.1'
    elif _data_var == '_Fpar_500m':
        scaling_factor = '0.01'
    else:
        scaling_factor = scaling_factor

    # multiply by scaling factor
    ds = ds*float(scaling_factor)
    LOG.info(f'multiplying by the scale factor of {scaling_factor}')

    base, extension = os.path.splitext(os.path.basename(path))

    path_analytics = create_dir(site_directory + '/MODIS/', 'timeSeries')
    
    # period dictionary in days according to the data_var
    # period = 365/temporal resolution of the layer 
    
    if detrend:
        
        new_fname = f"{base}.descaled.detrended{extension}"
        output_fname = f"{path_analytics}/{new_fname}"

        # detrend ts 
        trend_ds = ds[_data_var].rolling(time=int(period), min_periods=1, center=True).mean()
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


def main(site_directory, value):

    config = glob.glob(site_directory + f'/*_config.yml') 
    config_fname = config[0]
    start_date, end_date, products = read_config(config_fname)
    
    for i, j in enumerate(products):
    
        product = j['product']

        if product == 'MCD64A1.061':
            continue
    
        _data_var_list = j['data_var'] 

        for _data_var in _data_var_list:

            scaling_factor, s, smoothing_method, period = (
                j['scaling_factor'], j['smooth_factor'], j['smooth_method'], j['period'])
            pattern = (
                    site_directory +
                    f'MODIS/{product}/*/*/interpolated/*.{_data_var}.linear.{smoothing_method}.{s}.tif')
            path = glob.glob(pattern)[0] 
            
            LOG.info(f'Processing this file {path}')

            pixel_ts(path, site_directory, _data_var, scaling_factor, period, detrend=value)

    
if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <site_directory> <True/False>")  # the user has to input two arguments
    else:
        site_directory = sys.argv[1]
        value = sys.argv[2].lower()
        
        if value == 'true':
            value = True
        elif value == 'false':
            value = False
        else:
            print("Invalid value! Please enter 'True' or 'False'.")
            sys.exit(1)
        
        main(site_directory, value)

# example of user input arguments
# python pixel_ts.py /wp_data/sites/Norfolk/ True   
