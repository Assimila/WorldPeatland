import os.path
import yaml
import logging
from save_xarray_to_gtiff_old import *
from gdal_sheep import *
from MLEO_NN import *
from smoothn import smoothn
import re
import sys
sys.path.insert(0, '/workspace/WorldPeatland/code/')

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_file_name(file_path):
    
    """
    get_file_name from the file path returns the modis data product name
    and the version 
    
    INPUT
        - file_path (str) - it would be the one set by the user when running the downloader_wp
            + MODIS the path specific to download MODIS data
    OUTPUT
        - file_name[0] (str) - in this case it would be the MODIS data product name
        - file_name[1] (str) - in this case it would be the MODIS data product version
    """
    
    file_path_components = file_path.split('/')
    file_name = file_path_components[-1].rsplit('.', 1)
    return file_name[0], file_name[1]


def regrid_img(nested_list, target_img):
    """
    Regriding of all sentinel-2 tiff images to match B02 images with the highest spatial resolution,
    this function gives back
    """
    regrid_list = []
    for img in nested_list:
        # clipping the images to the shapefile created a problem later that not all images have the same extent
        # not able to stack them on top of each other probably because the NA values differ per date
        # thus, making the extent different depending on the available pixel values per day
        g = reproject_image(img[0], target_img)  # , clip_shapefile = shapefile_path, no_data_val = -9999)
        regrid_list.append(g)
    return regrid_list


def get_tif_path_dict(date, bands, S2_path, B02_path):
    """
    Get a dictionary with keys as band names and values is the regridded single tif file path for the selected month
    """

    # Create a nested list which is a list of lists each tiff corresponding to one band will be in a list
    # ==> the len of the nested list should be equal to the number of bands or subproducts
    nested = [sorted(glob.glob(f'{S2_path}/%s/*{date}.tif' % i))
              for i in bands]
    # can use code to choose a random B2 file as target file?
    # Target image here is a sentinel-2 B2 image -------  it has the highest resolution
    # pick a random B2 image as target_img
    target_img = glob.glob(f'{B02_path}/*.tif')[0]

    regrid_list = regrid_img(nested, target_img)
    # Create a dictionary from the nested list because the nested list doesn't keep record of the name of the bands
    # ==> to keep track of the name of the band for each tiff file we put them in a dictionary
    # Key:name of the bands
    # Values: list of the tiff files
    regrid_dict = dict(zip(bands, regrid_list))

    return regrid_dict


def create_ds(regrid_dict, bands):
    
    """
    create_ds will generate an xarray ds of all the sentinel 2 datasets and cloudmask
        
    OUTPUT:
        - ds (xarray.Dataset) - it contains all 15 sentinel bands plus the acm cloud mask 
            of all the files in the output_dir/Sentinel (total number of variables is 16)
    """

    stack, dts, opn = gdal_stack_dt(regrid_dict, 'time')

    stack_list = []
    for i in regrid_dict:
        stack, dts, opn = gdal_stack_dt(regrid_dict[i], 'time')
        stack_list.append(stack)
    stack_dict = dict(zip(bands, stack_list))

    xs, ys = create_coord_list(opn)

    ds = xr.Dataset(data_vars={i: (('time', 'latitude', 'longitude'), stack_dict[i])for i in bands},
                    coords={'time': dts, 'latitude': ys, 'longitude': xs})
    
    return ds, dts, ys, xs


def create_dir(output_dir, directory):
    
    """
    create_dir function will first check if the directory already exist if not it will
    create a directory where it will store the data to be downloaded
    
    INPUTS:
        - output_dir (str/path) - specified by the user where they want the data to be downloaded
        - directory (str) - specified by each step in the code to create,
        usually its the name of the data product to be downloaded
    """

    # Path 
    path = os.path.join(output_dir, directory) 
    
    if not os.path.exists(path):
        os.makedirs(path)
        LOG.info(f"Directory '{path}' created successfully.")
    else:
        LOG.info(f"Directory '{path}' already exists.")

    return path


def read_config(config_fname):
    """
    Read downloaders config file - Gerardo Saldana
    """
    with open(config_fname) as f:
        data = yaml.full_load(f)
    
    # Information from the first list index 0 about the site
    start_date = data[0]['start_date']
    end_date = data[0]['end_date']
    
    # Information about the first EO data product to download list index 1 
    products = data[1]['products']
    return start_date, end_date, products


def process_MLEONN(data, config_fname, dts, ys, xs, saved_path, tile_name, data_product):
    
    start_date, end_date, products = read_config(config_fname)
    
    # 1.Smooth data using smoothn
    smoothed_data = smoothn(y=data, s=10, isrobust=True, axis=0)[0]

    # 2.Create xarray to be able to interpolate in function of time 
    ds = xr.Dataset(data_vars={f'{data_product}_smooth': (('time', 'latitude', 'longitude'), smoothed_data)},
                    coords={'time': dts, 'latitude': ys, 'longitude': xs})
    
    # 3.Perform linear interpolation
    ds_linear = ds.interp(coords={'time': pa.date_range(start_date, end_date, freq='1D')}, method='linear')
    
    # 4.Set CRS attribute
    # get the proj4 str from the tif
    proj4_utm = '+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs'
    ds_linear.attrs['crs'] = proj4_utm
    
    # 5.Save as utm TIFF
    output_utm = os.path.join(saved_path, f'{data_product}_{tile_name}_smoothn_utm.tif')
    save_xarray_old(output_utm, ds_linear, f'{data_product}_smooth')

    # 6.If data is LAI, resample to 10 by 10 pixel size
    proj4_string = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'
    # change projection from utm to sinusoidal 
    output_sinu = os.path.join(saved_path, f'{data_product}_{tile_name}_smoothn_sinusoidal_resampled.tif')
    ds = gdal.Open(output_utm)

    # reproject to sinusoidal and resample to 10 by 10 pixel size
    dsReprj = gdal.Warp(output_sinu, ds, dstSRS=proj4_string, xRes=10, yRes=10)
    ds = dsReprj = None  # close the files
    LOG.info(f'{data_product}_resampled and saved')
        
    # 8.Delete UTM files
    os.remove(output_utm)


def get_date_list(B02_path):
    """
    get a list of the dates of the downloaded tif files chosen B02 just as a reference can be any band
    """
    # Get all the tiff files in the B02 folder
    tif_files = glob.glob(os.path.join(B02_path, '*.tif'))

    # Create an empty list to store the dates
    date_list = []

    # Loop over tif files list and extract the date
    for tif in tif_files:
        # Extract single tif file name from path
        filename = os.path.basename(tif)
        # Match using regex pattern in filename
        match = re.search(r'\d{4}-\d{2}', filename)  # date YYYY (4 digits) and MM (2 digits)
        # If match exists append
        if match:
            date_list.append(match.group())  # group transforms match regex object to string.

    return date_list


def main(S2_path):
    
    # tile_name, _ = get_file_name(geojson_path)

    # Path to B02 datacube tiff files
    B02_path = os.path.join(S2_path, 'B02')
    # Extract from tiff files' name year and month
    date_list = get_date_list(B02_path)

    # Get bands
    bands = os.listdir(S2_path)
    if 'SCL' in bands:
        bands.remove('SCL')

    for date in date_list:
        # get the dict of bands with the corresponding tif paths for the specific date
        regrid_dict = get_tif_path_dict(date, bands, S2_path, B02_path)
        # create an xarray for the month containing all bands
        ds, dts, ys, xs = create_ds(regrid_dict, bands)

        # apply mask before scale factor to not change the acm_mask 1 and 999 values
        # Set a threshold for cloud probability
        # MSK_CLDPRB valid values from 0 to 100 (100 % probability of pixel being cloudy)
        threshold = 20
        ds_masked = ds.where(ds['MSK_CLDPRB_20m'].values <= threshold, np.nan)

        # Multiply by the scaling factor of sentinel 2 data reflectance
        ds_masked = (ds_masked * 0.0001).astype(np.float32)

        # LAI_evaluatePixelOrig only takes a dictionary as input
        # thus, have to put ds_masked in a dictionary
        # band_names as key and values are the corresponding arrays ds_masked[band_name].values
        l = []
        for i in bands:
            li = ds_masked[i].values
            l.append(li)

        dict_ = dict(zip(bands, l))
        # TODO create numpy arrays of the angles

        # TODO add the numpy arrays of the angles to the dict_
        # save to netcdf file the masked sentinel-2 xarray
        saved_path = create_dir(S2_path, 'timeSeries')
        ds_masked.to_netcdf(f'{saved_path}/s2_masked_{date}.nc')

        lai = LAI_evaluatePixelOrig(dict_)
        fapar = FAPAR_evaluatePixelOrig(dict_)
        fc = FC_evaluatePixel(dict_)
        cab = CAB_evaluatePixel(dict_)

    # get config file name for the site and read start and end date
    config_fname = glob.glob(f'{output_dir}{tile_name}/*.yml')[0]
    process_MLEONN(lai, config_fname, dts, ys, xs, saved_path, tile_name, 'lai')
    process_MLEONN(fapar, config_fname, dts, ys, xs, saved_path, tile_name, 'fapar')
    process_MLEONN(fc, config_fname, dts, ys, xs, saved_path, tile_name, 'fc')
    process_MLEONN(cab, config_fname, dts, ys, xs, saved_path, tile_name, 'cab')


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <S2_path>")  # the user has to input one argument
    else:
        # location of the second item in the list
        # Sentinel 2 folder path where all bands folders are
        S2_path = sys.argv[1]
        main(S2_path)
