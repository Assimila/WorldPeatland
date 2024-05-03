from osgeo import gdal, osr
import numpy as np
import xarray as xr
import glob
import yaml
import logging
import os
from pathlib import Path
from datetime import datetime as dt

import sys
sys.path.insert(0,'/home/ysarrouh/')
from save_xarray_to_gtiff_old import *

from gdal_sheep import *
from MLEO_NN import *
from smoothn import smoothn



logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_file_name(file_path):
    
    '''get_file_name from the file path returns the modis data product name
    and the version 
    
    INPUT
        - file_path (str) - it would be the one set by the user when running the downloader_wp
            + MODIS the path specific to download MODIS data
    OUTPUT
        - file_name[0] (str) - in this case it would be the MODIS data product name
        - file_name[1] (str) - in this case it would be the MODIS data product version
    '''
    
    file_path_components = file_path.split('/')
    file_name = file_path_components[-1].rsplit('.', 1)
    return file_name[0], file_name[1]

def create_ds(output_dir, tile_name):
    
    '''
    create_ds will generate an xarray ds of all the sentinel 2 datasets and cloudmask
    
    INPUTS: 
        - output_dir (str) - where the initial MODIS and sentinel data were downloaded
        - tile_name (str) - obtained from the geojson 
        
    OUTPUT:
        - ds (xarray.Dataset) - it contains all 15 sentinel bands plus the acm cloud mask 
            of all the files in the output_dir/Sentinel (total number of variables is 16)
    '''

    # can use code to choose a random B2 file as target file?
    # Target image here is a sentinel-2 B2 image -------  it has the higehst resolution
    # pick a random B2 image as target_img
    target_img = glob.glob(f'{output_dir}/{tile_name}/Sentinel/datacube/S2_SR/B2/{tile_name}/*.tif')[0]

    # sub products
    # need a list of all the subproducts name in the datacube to keep a record of the product name in the dictionary
    # and later in the xarray variables
    bands = os.listdir(f'{output_dir}/{tile_name}/Sentinel/datacube/S2_SR/')

    acm_path = sorted(glob.glob(f'{output_dir}/{tile_name}/Sentinel/datacube/ACM/cloudmask/{tile_name}/*.tif'))

    # Create a nested list which is a list of lists each tiff corresponding to one band will be in a list 
    # ==> the len of the nested list should be equal to the number of bands or subproducts 
    nested = [sorted(glob.glob(f'{output_dir}/{tile_name}/Sentinel/datacube/S2_SR/%s/{tile_name}/*.tif'%i)) for i in bands]
    nested.append(acm_path)

    # name the variable to use it when adding it to the xarray 
    mask_var = 'acm_mask'

    # add the mask new name into the list of band names
    bands.append(mask_var)

    # Create a dictionary from the nested list because the nested list doesn't keep record of the name of the bands
    # ==> to keep track of the name of the band for each tiff file we put them in a dictionary 
    # Key:name of the bands
    # Values: list of the tiff files
    fdict = dict(zip(bands, nested))

    # sentinel-2 files has to be reprojected
    reprojected_list = []
    for i in fdict:
        band_list = []
        for img in fdict[i]:

            # clipping the images to the shapefile created a problem later that not all images have the same extent 
            # not able to stack them on top of each other probably because the NA values differ per date
            # thus, making the extent different depending on the available pixel values per day 
            g = reproject_image(img, target_img) #, clip_shapefile = shapefile_path, no_data_val = -9999)
            band_list.append(g)

        reprojected_list.append(band_list)

    reprojected_dict = dict(zip(bands, reprojected_list))

    stack_list = []
    for i in reprojected_dict:
        stack, dts, opn = gdal_stack_dt(reprojected_dict[i], 'time')
        stack_list.append(stack)
    stack_dict = dict(zip(bands, stack_list))

    xs, ys = create_coord_list(opn)
    variable_name = bands

    ds = xr.Dataset(data_vars = {i:(('time','latitude', 'longitude'),stack_dict[i])for i in bands},
                   coords={'time': dts,
                          'latitude': ys,
                          'longitude': xs})
    
    return ds, bands, dts, ys, xs

def create_dir(output_dir, directory):
    
    '''
    create_dir fucntion will first check if the directory already exist if not it will 
    create a directory where it will store the data to be downloaded
    
    INPUTS:
        - output_dir (str/path) - specified by the user where they want the data to be downloaded
        - directory (str) - specified by each step in the code to create, usually its the name of the data product to be downloaded
    '''

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


def process_MLEONN(data, config_fname, dts, ys, xs, saved_path, tile_name, data_type):
    
    start_date, end_date, products = read_config(config_fname)
    
    # 1.Smooth data using smoothn
    smoothed_data = smoothn(y=data, s=10, isrobust=True, axis=0)[0]

    # 2.Create xarray to be able to interpolate in function of time 
    ds = xr.Dataset(data_vars = {f'{data_type}_smooth':(('time','latitude', 'longitude'), smoothed_data)},
                   coords={'time': dts,
                          'latitude': ys,
                          'longitude': xs})
    
    # 3.Perform linear interpolation
    ds_linear = ds.interp(coords={'time': pa.date_range(start_date, end_date, freq='1D')}, method='linear')
    
    # 4.Set CRS attribute
    proj4_utm = '+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs'
    ds_linear.attrs['crs'] = proj4_utm
    
    # 5.Save as utm TIFF
    output_path = os.path.join(saved_path, f'{data_type}_{tile_name}_smoothn_utm.tif')
    save_xarray_old(output_path, ds_linear, f'{data_type}_smooth')

    # 6.If data is LAI, resample to 10 by 10 pixel size
    proj4_string = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'
    # change projection from utm to sinusoidal 
    output_raster = os.path.join(saved_path, f'{data_type}_{tile_name}_smoothn_sinusoidal_resampled.tif')
    ds = gdal.Open(output_path)

    # reproject to sinusoidal and resample to 10 by 10 pixel size
    dsReprj = gdal.Warp(output_raster, ds, dstSRS=proj4_string, xRes = 10, yRes = 10)
    ds = dsReprj = None # close the files
    LOG.info(f'{data_type}_resampled and saved')
        
    
    # 8.Delete UTM files
    os.remove(output_path)


def main(geojson_path, output_dir):
    
    tile_name,_ = get_file_name(geojson_path)
    
    ds, bands, dts, ys, xs= create_ds(output_dir, tile_name)
    
    # apply mask before scale factor to not change the acm_mask 1 and 999 values
    ds_masked = ds.where(ds['acm_mask'].values != 1, np.nan)
    
    # mutltiply by the scaling factor of sentinel 2 data reflectance
    ds = ds * 0.0001
    
    # LAI_evaluatePixelOrig only takes a dictionary as input
    # thus, have to put ds_masked in a dictionary 
    # band_names as key and values are the corresponding arrays ds_masked[band_name].values
    l = []
    for i in bands:
        li = ds_masked[i].values
        l.append(li)

    dict_ = dict(zip(bands, l))
    
    # save to netcdf file the masked sentinel-2 xarray 
    saved_path = create_dir(f'{output_dir}{tile_name}/Sentinel/', 'timeSeries')
    ds_masked.to_netcdf(f'{saved_path}/s2_masked.nc')
    
    LOG.info(f'masked sentinel 2 are saved here {saved_path}')
    
    # get config file name for the site and read start and end date
    config_fname = glob.glob(f'{output_dir}{tile_name}/*.yml')[0]
    
    lai = LAI_evaluatePixelOrig(dict_)
    process_MLEONN(lai, config_fname, dts, ys, xs, saved_path, tile_name, 'lai')
    
    fapar = FAPAR_evaluatePixelOrig(dict_)
    process_MLEONN(fapar, config_fname, dts, ys, xs, saved_path, tile_name, 'fapar')
    
    fc = FC_evaluatePixel(dict_)
    process_MLEONN(fc, config_fname, dts, ys, xs, saved_path, tile_name, 'fc')
    
    cab = CAB_evaluatePixel(dict_)
    process_MLEONN(cab, config_fname, dts, ys, xs, saved_path, tile_name, 'cab')
    
    
if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <geojson_path> <output_dir>") # the user has to input two arguments  
    else:
        geojson_path = sys.argv[1] # location of the second item in the list which is the first argument geojson site location 
        output_dir = sys.argv[2] # location of output downloaded data 3rd item in the list which is the 2nd argument
        main(geojson_path, output_dir)