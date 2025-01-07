import os.path
import glob
import re
from datetime import datetime as dt
import rasterio
import sys
import pandas as pd
import yaml
import numpy as np
import xarray as xr
import subprocess
from osgeo import gdal
import logging

from WorldPeatland.code.gdal_sheep import (gdal_stack_dt, create_coord_list, gdal_dt, create_xarr, reproject_image,
                                           get_proj4_from_tif)
from WorldPeatland.code.MLEO_NN import (LAI_evaluatePixelOrig, FAPAR_evaluatePixelOrig, FC_evaluatePixel,
                                        CAB_evaluatePixel)
from WorldPeatland.code.smoothn import smoothn
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old, save_tiff


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


def create_ds(regrid_dict, bands):
    """
    create_ds will generate a xarray ds of all the sentinel 2 datasets and cloudmask
    INPUTS:
        - bands (list) - keys of the dictionary with name of the bands
    OUTPUT:
        - ds (xarray.Dataset) - it contains all 15 sentinel bands plus the acm cloud mask 
            of all the files in the output_dir/Sentinel (total number of variables is 16)
    """

    stack, dts, opn = gdal_stack_dt(regrid_dict)

    stack_list = []
    for i in regrid_dict:
        stack, dts, opn = gdal_stack_dt(regrid_dict[i])
        stack_list.append(stack)
    stack_dict = dict(zip(bands, stack_list))

    xs, ys = create_coord_list(opn)

    ds = xr.Dataset(data_vars={i: (('time', 'latitude', 'longitude'), stack_dict[i]) for i in bands},
                    coords={'time': dts, 'latitude': ys, 'longitude': xs})

    return ds, dts, ys, xs


def create_dir(output_dir, directory):
    """
    create_dir function will first check if the directory already exist if not it will
    create a directory where it will store the data to be downloaded
    
    INPUTS:
        - output_dir (str/path) - specified by the user where they want the data to be downloaded
        - directory (str) - specified by each step in the code to create,
        usually it's the name of the data product to be downloaded
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

    # 1.Smooth data using smoothn can smoothn all the biophysical parameters
    # use dask concatenate
    smoothed_data = smoothn(y=data, s=10, isrobust=True, axis=0)[0]

    # 2.Create xarray to be able to interpolate in function of time 
    ds = xr.Dataset(data_vars={f'{data_product}_smooth': (('time', 'latitude', 'longitude'), smoothed_data)},
                    coords={'time': dts, 'latitude': ys, 'longitude': xs})

    # 3.Perform linear interpolation
    ds_linear = ds.interp(coords={'time': pd.date_range(start_date, end_date, freq='1D')}, method='linear')

    # 4.Set CRS attribute
    # TODO get the proj4 str from the tif
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


def get_timestep_from_tif(tif):
    """
    Extract the timestep (date) from a geotif filename

    INPUT
        - tif (string) - tif file path where the filename ends with for example *2018-06.*extension*

    OUTPUT
        - timestep (string) - example 2018-06 (YYYY-MM)
    """

    # Search for the same file but for all other reflectance bands
    # get the date from tif file name
    filename = os.path.basename(tif)
    # Match using regex pattern in filename
    match = re.search(r'\d{4}-\d{2}', filename)  # date YYYY (4 digits) and MM (2 digits)

    return match.group()


def get_bands_SR_in_arrays(bd, tif, opn, month_tif, input_dict):
    """
    1. Get the surface reflectance bands (other than B2 SR)
    2. regrid to 10 m pixel resolution
    3. put the surface reflectance in arrays + multiply by scaling factor
    4. add to input_dict

    OUTPUT
        - input_dict (dictionary) - populated input_dict with all surface reflectance bands needs for MLEONN
    """

    # get name of the spectral band from the tif path
    band_name = tif.split('/')[-2]

    # Check that all monthly tif files for all the reflectance have the same no. of timesteps as B02
    with rasterio.open(tif) as src:
        if src.count != opn.RasterCount:
            raise ValueError(f'Raster count for {tif} does not match B02 timesteps'
                             f'Expected: {opn.RasterCount}, Found: {src.count}')

    # regrid the tifs for the same B02 pixel size
    g = reproject_image(tif, month_tif)
    # g is a gdal.dataset, pick the same raster band number in the loop
    rb_g = g.GetRasterBand(bd)
    arr_g = rb_g.ReadAsArray()
    SR_BD_SCALE = 10000
    input_dict[f'{band_name}'] = (arr_g / SR_BD_SCALE).astype(np.float32)

    return input_dict


def prepare_run_MLEONN(bd, opn, month_tif, dts, reflectance_tifs, MLEONN_products_dict):
    """
    Prepare the input bands surface reflectance + Run MLEONN
    OUTPUT
        - MLEONN_products_dict (dictionary) - dict with keys as MLEONN products name and with
            corresponding values as list of arrays
    """
    # Start by adding the reflectance B02 array to the input_dictionary
    rb = opn.GetRasterBand(bd)
    # Read the band 2d array
    arr = rb.ReadAsArray()
    # multiply by scaling factor
    SR_BD_SCALE = 10000
    arr = (arr / SR_BD_SCALE).astype(np.float32)
    # create a dictionary where you will append all the 2d arrays needed as inputs for MLEONN
    input_dict = {'B02': arr}
    # Now loop over all other bands
    for tif in reflectance_tifs:
        # Populate the input_dict with descaled data from the other bands (other than B2)
        input_dict = get_bands_SR_in_arrays(bd, tif, opn, month_tif, input_dict)

    # Get the in band metadata information from the B02 monthly tif file
    mtd = rb.GetMetadata()  # mtd is a dictionary containing all in band metadata

    # Get the timestep from the metadata to reconstruct the monthly tif of the MLEONN products
    time_str = mtd['time']
    dt_format = '%Y-%m-%d %H:%M:%S'
    time = dt.strptime(time_str, dt_format)
    dts.append(time)

    # extract the angular information as numpy float 32 (it is set as str)
    vaa = np.float32(mtd['vaa'])
    vza = np.float32(mtd['vza'])
    saa = np.float32(mtd['saa'])
    sza = np.float32(mtd['sza'])

    # add the angular information into a dictionary
    angular_dict = {'vaa': vaa, 'vza': vza, 'saa': saa, 'sza': sza}

    # form a numpy array for each angle same shape as the B02 reflectance array
    vaa = np.full(arr.shape, vaa)
    vza = np.full(arr.shape, vza)
    saa = np.full(arr.shape, saa)
    sza = np.full(arr.shape, sza)

    # add the arrays to the input_dict
    input_dict['vaa'] = vaa
    input_dict['vza'] = vza
    input_dict['saa'] = saa
    input_dict['sza'] = sza

    # process MLEONN as one timestep per raster band
    lai = LAI_evaluatePixelOrig(input_dict)
    fapar = FAPAR_evaluatePixelOrig(input_dict)
    fc = FC_evaluatePixel(input_dict)
    cab = CAB_evaluatePixel(input_dict)

    # Empty the input_dict
    input_dict = None

    # Append the MLEONN product arrays to the dictionary
    MLEONN_products_dict['lai'].append(lai)
    MLEONN_products_dict['fapar'].append(fapar)
    MLEONN_products_dict['fc'].append(fc)
    MLEONN_products_dict['cab'].append(cab)
    LOG.info(f'MLEONN four products successfully formed')
    return MLEONN_products_dict, angular_dict


def create_monthly_cogs(product, S2_path, timestep, opn, dts, month_tif, MLEONN_products_dict):
    """
    Create cog monthly tiffs for the MLEONN products created
    1. stack the arrays all the days available with the month
    2. Create a xarray
    3. Get the crs proj-4 fromt the downloaded initial S2 B2 monthly tif
    4. save the xarray as tif
    """

    # loop over all MLEONN generated products to save them to cog monthly tiffs

    # stack the numpy arrays in the lists
    stacked_arr = np.stack(MLEONN_products_dict[f'{product}'], axis=0)

    # create name of the output directory for the MLEONN products
    output_dir = create_dir(os.path.dirname(S2_path), f'MLEONN/{product}')

    # Output file name
    output_cog_fname = f'{product}_{timestep}.tif'
    output_cog_fname = os.path.join(output_dir, output_cog_fname)

    # create an xarray for each MLEONN product
    x_array = create_xarr(opn, product, stacked_arr, dts)

    # get crs or the proj4 str from the monthly-tif using the command line of gdalinfo
    command = ['gdalinfo', month_tif, '-proj4']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # use regex to find the string in the results
    proj4_match = re.search(r"PROJ\.4 string is:\n'(.*?)'", result.stdout)
    proj4_string = proj4_match.group(1)
    # set proj4 str as an attribute to the xarray so that it can be saved as a tiff
    x_array.attrs['crs'] = proj4_string

    # edited version of save_xarray_to_gtiff
    save_tiff(output_cog_fname, x_array, product)
    LOG.info(f'COG successfully saved {output_cog_fname}')

    return output_cog_fname


def directional_distance_transform(is_cloud, saa, max_distance):
    """
    Projects distances from clouds in a specific direction.

    Parameters:
    is_cloud (np.ndarray): Binary mask where clouds are marked with 1 and non-clouds with 0.
    saa (float): Solar Azimuth Angle - direction in degrees (0-360) where 0 is North.
    max_distance (int): Maximum distance to project.

    Returns:
    np.ndarray: Distance transform array.
    """
    # Convert azimuth to radians
    azimuth_rad = np.deg2rad(90 - saa)

    # Calculate the direction vector
    dx = np.sin(azimuth_rad)
    dy = np.cos(azimuth_rad)

    # Initialize the distance transform array
    distance_transform = np.full(is_cloud.shape, np.inf)

    # Get the indices of cloud pixels
    cloud_indices = np.argwhere(is_cloud.data == 1)

    for y, x in cloud_indices:
        for d in range(1, max_distance + 1):
            # Calculate the new position
            new_x = int(round(x + d * dx))
            new_y = int(round(y + d * dy))

            # Check if the new position is within bounds
            if 0 <= new_x < is_cloud.shape[1] and 0 <= new_y < is_cloud.shape[0]:
                distance_transform[new_y, new_x] = min(distance_transform[new_y, new_x], d)

    # Replace inf with 0 for non-cloud areas
    distance_transform[np.isinf(distance_transform)] = 0

    return distance_transform


def main(site_dir):

    s2_path = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR')
    LOG.info(f'Start of Processing of {s2_path}')

    # Path to B02 datacube tiff files
    pattern = os.path.join(s2_path, 'B2', '*.tif')
    # get B02 monthly tif files
    B2_tif_files_list = glob.glob(pattern)

    # Set reflectance bands list
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    for month_tif in B2_tif_files_list:
        # extract timestep from the filename
        timestep = get_timestep_from_tif(month_tif)
        LOG.info(f'Starting processing this timestep: {timestep}')
        # list of all SR bands tiffs for the same month
        reflectance_tifs = []
        for band in bands:
            pattern = os.path.join(s2_path, band, f'*{timestep}*.tif')
            reflectance_tifs.extend(glob.glob(pattern, recursive=True))

        # Open the monthly B02 tif file
        opn = gdal.Open(month_tif)

        # Check that all monthly tif files for all the reflectance have the same no. of timesteps as B02
        for tif in reflectance_tifs:
            with rasterio.open(tif) as src:
                if src.count != opn.RasterCount:
                    raise ValueError(f'Raster count for {tif} does not match B02 timesteps'
                                     f'Expected: {opn.RasterCount}, Found: {src.count}')

        # MLEONN products name list
        MLEONN_products_name = ['lai', 'fapar', 'fc', 'cab']
        # Empty dictionary to store the biophysical MLEONN products
        MLEONN_products_dict = {}
        for product in MLEONN_products_name:
            MLEONN_products_dict[product] = []
        # Datetime empty list
        dts = []
        # Create an empty list to append the in month single days saa
        angular_list = []

        # Now loop over each timestep in this month, loop over each raster in the tif
        # in this case bd as band meaning a 2d raster or one timestep and not reflectance bands
        for bd in range(1, opn.RasterCount + 1):  # gdal starts raster band count from 1
            # PREPARE INPUT BAND SR AND RUN MLEONN
            LOG.info('MLEONN process starting')
            MLEONN_products_dict, angular_dict = prepare_run_MLEONN(bd, opn, month_tif, dts, reflectance_tifs,
                                                                    MLEONN_products_dict)
            angular_list.append(angular_dict)

        # empty dictionary  to store outputs file names
        outputs = {}
        for product in MLEONN_products_name:
            output_cog_fname = create_monthly_cogs(product, s2_path, timestep,
                                                   opn, dts, month_tif, MLEONN_products_dict)

            # If the product key is not already in the outputs dictionary, initialize it with an empty list
            if product not in outputs:
                outputs[product] = []
            # Append the output filename to the corresponding product key
            outputs[product].append(output_cog_fname)

        # Create cloud mask

        LOG.info(f'Cloud mask processing started')
        # Set thresholds
        CLD_PRB_THRESH = 0.5
        # NIR dark pixel reflectance threshold is set to 0.15 already descaled
        NIR_DRK_THRESH = 0.25
        maxDis = 150

        # Reproject SCL layer once per timestep
        pattern = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR', 'SCL', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]
        # Resample the SCL band (downscaling the SCL band pixel originally 60m to 20m)
        SCL_gdalobj = reproject_image(fname, month_tif)
        SCL_resampled = SCL_gdalobj.ReadAsArray()  # array dtype=uint8
        water_class = 6
        water_pixels = (SCL_resampled == water_class)

        if water_pixels.ndim == 2:
            water_pixels = water_pixels[np.newaxis, :]

        water_pixels = create_xarr(opn, 'boolean', water_pixels, dts)

        pattern = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR', 'B8', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]

        arr, dts, saved_opn = gdal_dt(fname)
        b8 = create_xarr(opn, 'boolean', arr, dts)
        b8 = b8 / 10000.0  # apply scaling factor

        pattern = os.path.join(site_dir, 'MSIL1C', 'datacube', 'S2_TOA', 'cprob', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]
        arr, dts, saved_opn = gdal_dt(fname)
        cloud_probability = create_xarr(opn, 'boolean', arr, dts)

        is_cloud = (cloud_probability > CLD_PRB_THRESH).astype(bool)
        dark_pixels = (b8 < NIR_DRK_THRESH).astype(bool)

        # get angular information from B2 in band metadata
        pattern = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR', 'B2', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]

        dataset = gdal.Open(fname)
        band = dataset.GetRasterBand(bd)
        metadata = band.GetMetadata()  # metadata is a dict
        saa = metadata.get("saa", "No 'saa' metadata found")
        saa = float(saa)

        distance_transform = (directional_distance_transform(is_cloud.boolean.values, saa, maxDis) > 0).astype(bool)
        shadows = dark_pixels * distance_transform

        # Create cloud mask logic using the water_pixels for the current inband timestep
        cloud_mask = (is_cloud | shadows) & ~water_pixels
        LOG.info(f'Cloud mask successfully formed for {timestep}')

        # Add crs to xarray
        proj4_string = get_proj4_from_tif(month_tif)
        cloud_mask.attrs['crs'] = proj4_string
        path = os.path.join(site_dir, 'MSIL2A', 'datacube', 'MLEONN')
        output_dir = create_dir(path, f'cloud_mask')
        LOG.info(f'Saving tif here: {output_dir}')
        fname_cloud_mask = os.path.join(output_dir, f'cloud_mask_{timestep}.tif')
        save_xarray_old(fname_cloud_mask, cloud_mask, f'boolean')
        LOG.info(f'tif successfully saved!')

        # Apply mask can be done after creating MLEONN products
        # # apply mask before scale factor to not change the acm_mask 1 and 999 values
        # # Set a threshold for cloud probability
        # # MSK_CLDPRB valid values from 0 to 100 (100 % probability of pixel being cloudy)
        # threshold = 20
        # ds_masked = ds.where(ds['MSK_CLDPRB_20m'].values <= threshold, np.nan)
        #
        #     # # create a xarray for the month containing all bands
        #     ds, dts, ys, xs = create_ds(input_dict, band_names)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <site_dir>")  # the user has to input one argument
    else:
        # location of the second item in the list
        # Sentinel 2 folder path where all bands folders are
        site_dir = sys.argv[1]
        main(site_dir)
