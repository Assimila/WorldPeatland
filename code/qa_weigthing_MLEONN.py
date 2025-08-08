import os.path as osp
import sys
from glob import glob
import pandas as pd
import os
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr
import rioxarray
import rasterio
import osgeo.gdal as gdal
import logging
from WorldPeatland.code.utils import create_dir
from WorldPeatland.code.gdal_sheep import get_proj4_from_tif
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

site = sys.argv[1]

# MLEONN lai RMSE: 0.7 for validation
# Reference S2 ToolBox ATBD V2.0
lai = {'variable': 'lai',
       'uncertainty_variable':'lai_unc',
       'weighting_factor': 0.02,
       'nominal_uncert_data': 0.7}


def _get_times(tif_path):
    """
    Extract datetime64 values from per-band metadata in a GeoTIFF.
    """
    d = gdal.Open(tif_path)
    n_bands = d.RasterCount

    times = []

    for n_band in range(n_bands):
        b = d.GetRasterBand(n_band + 1)
        md = b.GetMetadata()
        times.append(md['time'])

    # Convert to pandas datetime Series and return as numpy array
    return pd.to_datetime(times, format='%Y-%m-%dT%H:%M:%S').to_numpy()


def process_large_dask_chunks(dask_datasets, block_size, nominal_uncert, unc_weight, save_dir):
    """
    Process large Dask datasets by splitting into time chunks, applying weighting, and saving results.
    """
    mleo_arr, mask_arr = dask_datasets
    mleo_arr = mleo_arr.astype('float32')

    latitude_chunks = range(0, mleo_arr.latitude.size, block_size)
    longitude_chunks = range(0, mleo_arr.longitude.size, block_size)

    # store paths to all blocks after smoothn1 + linear + smoothn2
    processed_blocks = []

    os.makedirs(save_dir, exist_ok=True)

    with ProgressBar():
        for lat_start in latitude_chunks:
            for lon_start in longitude_chunks:
                lat_end = min(lat_start + block_size, mleo_arr.latitude.size)
                lon_end = min(lon_start + block_size, mleo_arr.longitude.size)

                mleo_block = mleo_arr.isel(latitude=slice(lat_start, lat_end),
                                           longitude=slice(lon_start, lon_end))

                mask_block = mask_arr.isel(latitude=slice(lat_start, lat_end),
                                           longitude=slice(lon_start, lon_end))

                # Apply weighting where mask == 0
                weighted_block = xr.where(
                    mask_block == 0,
                    mleo_block * unc_weight,
                    xr.full_like(mleo_block, nominal_uncert)
                )

                result = weighted_block.compute()
                save_path = os.path.join(save_dir, f"{lat_start}_{lon_start}.weighted.nc")
                result.to_netcdf(save_path)

                processed_blocks.append(save_path)
                LOG.info(f'{save_path} successfully saved')

    logging.info('Combining all weighted uncertainty blocks into one xarray dataset')
    combined = xr.open_mfdataset(processed_blocks, chunks={"latitude": 256, "longitude": 256})
    logging.info('Weighted uncertainty blocks successfully combined')

    return combined


# TODO later on add the other mleonn products
products = [lai]

for _product in products:

    variable = _product['variable']
    var_name = _product['uncertainty_variable']
    weighting_factor = _product['weighting_factor']
    nominal_uncert_data = _product['nominal_uncert_data']

    # Paths to the interpolated and smoothed time series

    data_file = f"/wp_data/sites/{site}/Sentinel/MSIL2A/datacube/MLEONN/{variable}/interpolated/{variable}.linear.smoothn1.5.tif"
    LOG.info(f'Opening smoothed and interpolated data: {data_file}')
    data_file = glob(data_file)[0]

    # Open the mask monthly Geotif and put in one xarray
    mask_monthly_tifs = sorted(glob(f"/wp_data/sites/{site}/Sentinel/MSIL2A/datacube/MLEONN/mask/*.tif"))

    #  Load and prepare all datasets
    datasets = []
    all_times = []

    for fpath in mask_monthly_tifs:

        LOG.info(f'Opening mask: {fpath}')
        data = rioxarray.open_rasterio(fpath)
        data = data.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
        datasets.append(data)

        # Extract real time values from metadata
        times = _get_times(fpath)
        all_times.extend(times)

    # Concatenate a long time axis
    LOG.info(f'All mask monthly tifs are being concatinated')
    mask = xr.concat(datasets, dim='time')
    LOG.info(f'All mask monthly tifs sucessfully concatinated')

    # Replace dummy time with real datetime
    mask = mask.assign_coords(time=all_times)

    output_dir = f"/wp_data/sites/{site}/Sentinel/MSIL2A/datacube/MLEONN/{variable}/interpolated/"
    output_fname = osp.basename(data_file.replace(".tif", "_qa_weighted_std_dev.tif"))
    output_fname = osp.join(output_dir, output_fname)

    # Open the VRT file using rioxarray
    mleo_data = rioxarray.open_rasterio(data_file)
    mleo_data = mleo_data.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
    mleo_data = mleo_data.assign_coords(time=all_times)


    # Check if both layers have the same shape
    if mleo_data.shape == mask.shape:
        LOG.info("The data and mask have the same shape.")
    else:
        LOG.info("The data and mask do not have the same shape.")
        LOG.info(f"Data shape: {mleo_data.shape}")
        LOG.info(f"Mask shape: {mask.shape}")
        raise ValueError("Data and mask shapes do not match.")

    # Apply chunking before any computation
    block_size = 100
    mleo_data = mleo_data.chunk({'time': -1, "latitude": 5, "longitude": 5})
    mask = mask.chunk({'time': -1, "latitude": 5, "longitude": 5})

    LOG.info('Begin block processing]')
    output_blocks_dir = create_dir(f"/wp_data/sites/{site}/Sentinel/MSIL2A/datacube/MLEONN/{variable}/interpolated/",
                                   'dask_processing_outputs_qa')
    data_stddev_weighted = process_large_dask_chunks(
        dask_datasets=[mleo_data, mask],  # list of the chunked datasets needed to calculate weighted unc
        block_size=block_size,  # Temporal block size
        nominal_uncert=nominal_uncert_data,
        unc_weight=weighting_factor,  # weighting factor for inflating the uncertainty
        save_dir=output_blocks_dir,
    )

    data_stddev_weighted = data_stddev_weighted.astype('float32')
    data_stddev_weighted = data_stddev_weighted.assign_coords(time=all_times)
    data_stddev_weighted.attrs['_FillValue'] = np.nan

    # Add the new weighted variable stddev as a new variable in the dataset
    data_stddev_weighted = data_stddev_weighted.rename({'time': 'RANGEBEGINNINGDATE',
                                                        '__xarray_dataarray_variable__': var_name})
    get_proj4_from_tif(data_file, data_stddev_weighted)
    opn = gdal.Open(data_file)
    save_xarray_old(output_fname, data_stddev_weighted, var_name, opn.GetGeoTransform())

    LOG.info('Script process Done!')

    # Print confirmation
    LOG.info("Saved the weighted stddev as a Cloud Optimized GeoTIFF.")
