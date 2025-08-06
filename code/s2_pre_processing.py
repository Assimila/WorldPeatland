import os.path
from glob import glob
from datetime import datetime as dt
import rasterio
import sys
import numpy as np
from osgeo import gdal
import logging

from WorldPeatland.code.gdal_sheep import (gdal_dt, create_xarr, reproject_image,
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
    # fapar = FAPAR_evaluatePixelOrig(input_dict)
    # fc = FC_evaluatePixel(input_dict)
    # cab = CAB_evaluatePixel(input_dict)

    # Empty the input_dict
    input_dict = None
    LOG.info(f'MLEONN four products successfully formed for {time}')

    return lai, time # fapar, fc, cab,


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

    s2_path = os.path.join(site_dir, 'Sentinel', 'MSIL2A', 'datacube', 'S2_SR')
    LOG.info(f'Start of Processing of {s2_path}')

    # Path to B02 datacube tiff files
    pattern = os.path.join(s2_path, 'B02', '*.tif')
    # get B02 monthly tif files
    B02_tif_files_list = sorted(glob(pattern))

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
            reflectance_tifs.extend(glob(pattern, recursive=True))

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
            lai, specific_date = prepare_run_MLEONN(bd, opn, month_tif, reflectance_tifs) #fapar, fc, cab,
            dts.append(specific_date)
            lai_list.append(lai)
            # fapar_list.append(fapar)
            # fc_list.append(fc)
            # cab_list.append(cab)

        # stack the arrays
        lai_array = np.stack(lai_list, axis=0)
        # fapar_array = np.stack(fapar_list, axis=0)
        # fc_array = np.stack(fc_list, axis=0)
        # cab_array = np.stack(cab_list, axis=0)

        # Create mask
        LOG.info(f'Mask processing started')

        # Reproject SCL layer once per timestep
        pattern = os.path.join(site_dir, 'Sentinel', 'MSIL2A', 'datacube', 'S2_SR', 'SCL', f'*{timestep}.tif')
        fname = glob(pattern)[0]
        # Resample the SCL band (downscaling the SCL band pixel originally 60m to 20m)
        SCL_gdalobj = reproject_image(fname, month_tif)
        SCL_resampled = SCL_gdalobj.ReadAsArray()  # array dtype=uint8

        # Create masks for each class based on SCL
        is_water = (SCL_resampled == 6).astype(bool)
        is_snow = (SCL_resampled == 11).astype(bool)
        is_cirrus = (SCL_resampled == 10).astype(bool)
        is_cloud = ((SCL_resampled == 8) | (SCL_resampled == 9)).astype(bool)
        is_shadow = (SCL_resampled == 3).astype(bool)

        # Create cloud mask logic using the water_pixels for the current inband timestep
        # Get the numpy arrays of the xr.datasets to preserve shape
        # lai_array > 8.0 removing impossible values of LAI according to S2Toolbox Level2Products version 2.0
        # table 9.
        mask = (
                is_cloud | is_shadow | is_snow | is_cirrus |
                (lai_array > 8.0) | (lai_array < 0) | is_water
        ) # dtype numpy array
        # Construct file path and save
        proj4_string = get_proj4_from_tif(month_tif)

        path = os.path.join(site_dir, 'Sentinel', 'MSIL2A', 'datacube', 'MLEONN')
        output_dir = create_dir(path, 'mask')
        fname = os.path.join(output_dir, f"mask_{timestep}.tif")
        ds_mask = create_xarr(opn, 'bool', mask, dts)
        ds_mask.attrs['crs'] = proj4_string
        save_xarray_old(fname, ds_mask, 'bool')

        LOG.info(f'Mask successfully formed & saved for {timestep}')

        variables = {
            'lai': lai_array,
            # 'fapar': fapar_array,
            # 'fc': fc_array,
            # 'cab': cab_array
        }

        for varname, var_arr in variables.items():
            LOG.info(f"Processing {varname}...")
            create(var_arr, mask, dts, opn, varname, proj4_string, path, timestep)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <site_dir>")  # the user has to input one argument
    else:
        # location of the second item in the list
        # Sentinel 2 folder path where all bands folders are
        site_dir = sys.argv[1]
        main(site_dir)
