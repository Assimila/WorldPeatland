import os.path
import glob
import re
from datetime import datetime as dt
import rasterio
import sys
import yaml
import numpy as np
import xarray as xr
from osgeo import gdal
import logging

from WorldPeatland.code.gdal_sheep import (gdal_stack_dt, create_coord_list, gdal_dt, create_xarr, reproject_image,
                                           get_proj4_from_tif)
from WorldPeatland.code.MLEO_NN import (LAI_evaluatePixelOrig, FAPAR_evaluatePixelOrig, FC_evaluatePixel,
                                        CAB_evaluatePixel)
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from WorldPeatland.code.utils import create_dir, get_timestep_from_tif

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


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


def prepare_run_MLEONN(bd, opn, month_tif, reflectance_tifs):
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

    # extract the angular information as numpy float 32 (it is set as str)
    vaa = np.float32(mtd['vaa'])
    vza = np.float32(mtd['vza'])
    saa = np.float32(mtd['saa'])
    sza = np.float32(mtd['sza'])

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
    LOG.info(f'MLEONN four products successfully formed')

    return lai, fapar, fc, cab, time


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


def create(var_arr, cloud_mask, dts, opn, varname, proj4_string, path, timestep):
    """
    Processes and saves geospatial data with cloud masking applied.

    Parameters:
        var_arr (numpy.ndarray): Input data array for the variable (e.g., LAI, fapar).
        cloud_mask (numpy.ndarray): Boolean array indicating cloud-covered areas (True = cloud).
        dts (datetime): Timestamps associated with the data.
        opn (dict): Operational metadata or parameters.
        varname (str): Name of the variable being processed.
        proj4_string (str): CRS in Proj4 format.
        path (str): Directory path to save the output files.
        timestep (str): Timestep identifier for the output file name.

    Returns:
        None
    """
    try:
        # Apply cloud mask to the data array
        masked_arr = np.where(~cloud_mask, var_arr, np.nan)

        # Create an xarray object
        masked_xr = create_xarr(opn, varname, masked_arr, dts)
        masked_xr.attrs['crs'] = proj4_string

        # Create output directory
        output_dir = create_dir(path, varname)
        LOG.info(f"Saving GeoTIFF here: {output_dir}")

        # Construct file path and save
        fname = os.path.join(output_dir, f"{varname}_{timestep}.tif")
        save_xarray_old(fname, masked_xr, varname)

    except Exception as e:
        LOG.error(f"Error in creating file for {varname}: {e}")


def main(site_dir):

    # Check if site_dir exist
    if os.path.isdir(site_dir):
        LOG.info(f"The directory '{site_dir}' exists.")
    else:
        LOG.info(f"The directory '{site_dir}' does not exist.")

    s2_path = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR')
    LOG.info(f'Start of Processing of {s2_path}')

    # Path to B02 datacube tiff files
    pattern = os.path.join(s2_path, 'B02', '*.tif')
    # get B02 monthly tif files
    B02_tif_files_list = glob.glob(pattern)

    # Set reflectance bands list
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

    for month_tif in B02_tif_files_list:
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

        # Datetime empty list
        dts = []
        lai_list = []
        fapar_list = []
        fc_list = []
        cab_list = []

        # Now loop over each timestep in this month, loop over each raster in the tif
        # in this case bd as band meaning a 2d raster or one timestep and not reflectance bands
        for bd in range(1, opn.RasterCount + 1):  # gdal starts raster band count from 1
            # PREPARE INPUT BAND SR AND RUN MLEONN
            LOG.info(f'MLEONN process starting for {timestep}')
            lai, fapar, fc, cab, specific_date = prepare_run_MLEONN(bd, opn, month_tif, reflectance_tifs)
            dts.append(specific_date)
            lai_list.append(lai)
            fapar_list.append(fapar)
            fc_list.append(fc)
            cab_list.append(cab)

        # stack the arrays
        lai_array = np.stack(lai_list, axis=0)
        fapar_array = np.stack(fapar_list, axis=0)
        fc_array = np.stack(fc_list, axis=0)
        cab_array = np.stack(cab_list, axis=0)

        # Create cloud mask

        LOG.info(f'Cloud mask processing started')
        # Set thresholds
        CLD_PRB_THRESH = 0.5
        # NIR dark pixel reflectance threshold is set to 0.15 already descaled
        NIR_DRK_THRESH = 0.25
        maxDis = 500

        # Reproject SCL layer once per timestep
        pattern = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR', 'SCL', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]
        # Resample the SCL band (downscaling the SCL band pixel originally 60m to 20m)
        SCL_gdalobj = reproject_image(fname, month_tif)
        SCL_resampled = SCL_gdalobj.ReadAsArray()  # array dtype=uint8
        water_class = 6
        water_pixels = (SCL_resampled == water_class)  # dtype numpy array
        # Check if arr is 2D make it 3D to fit the time dimension
        if water_pixels.ndim == 2:
            water_pixels = water_pixels[np.newaxis, :]

        pattern = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR', 'B08', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]

        b8_arr, dts, saved_opn = gdal_dt(fname)
        # Check if arr is 2D make it 3D to fit the time dimension
        if b8_arr.ndim == 2:
            b8_arr = b8_arr[np.newaxis, :]

        b8_arr = b8_arr / 10000.0  # apply scaling factor

        pattern = os.path.join(site_dir, 'MSIL1C', 'datacube', 'S2_TOA', 'cprob', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]
        cloud_probability, dts, saved_opn = gdal_dt(fname)
        # Check if arr is 2D make it 3D to fit the time dimension
        if cloud_probability.ndim == 2:
            cloud_probability = cloud_probability[np.newaxis, :]

        is_cloud = (cloud_probability > CLD_PRB_THRESH).astype(bool)
        dark_pixels = (b8_arr < NIR_DRK_THRESH).astype(bool)

        # get angular information from B2 in band metadata
        pattern = os.path.join(site_dir, 'MSIL2A', 'datacube', 'S2_SR', 'B02', f'*{timestep}.tif')
        fname = glob.glob(pattern)[0]

        dataset = gdal.Open(fname)
        band = dataset.GetRasterBand(bd)
        metadata = band.GetMetadata()  # metadata is a dict
        saa = metadata.get("saa", "No 'saa' metadata found")
        saa = float(saa)

        distance_transform = (directional_distance_transform(is_cloud, saa, maxDis) > 0).astype(bool)
        shadows = dark_pixels * distance_transform
        # Create cloud mask logic using the water_pixels for the current inband timestep
        # Get the numpy arrays of the xr.datasets to preserve shape
        cloud_mask = (is_cloud | shadows) & ~water_pixels  # dtype numpy array
        LOG.info(f'Cloud mask successfully formed for {timestep}')

        variables = {
            'lai': lai_array,
            'fapar': fapar_array,
            'fc': fc_array,
            'cab': cab_array
        }

        proj4_string = get_proj4_from_tif(month_tif)
        path = os.path.join(site_dir, 'MSIL2A', 'datacube', 'MLEONN')
        for varname, var_arr in variables.items():
            LOG.info(f"Processing {varname}...")
            create(var_arr, cloud_mask, dts, opn, varname, proj4_string, path, timestep)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <site_dir>")  # the user has to input one argument
    else:
        # location of the second item in the list
        # Sentinel 2 folder path where all bands folders are
        site_dir = sys.argv[1]
        main(site_dir)
