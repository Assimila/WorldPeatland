import sys
import os
import xarray as xr
import logging
from glob import glob
from dask.diagnostics import ProgressBar
from WorldPeatland.code.utils import create_dir
from WorldPeatland.code.gdal_sheep import concat_xr, get_proj4_from_tif
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from WorldPeatland.code.s1_pre_processing import DW_smoothn_smooth_xarray

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def interpolate_block(block):
    """
    interpolate a single block and save it to disk.
    """
    LOG.info(f"Linear interpolation started for the block")
    interpolated_block = block.interpolate_na(dim='time', method='linear')
    LOG.info(f'Linear interpolation successful')
    return interpolated_block


def process_and_save_block(block, save_path, var_name, window_size, flag):
    """
    Smooth a single block and save it to disk.
    """
    LOG.info(f"Smoothn started for the block")
    smoothed_block = DW_smoothn_smooth_xarray(block, var_name, window_size, flag)

    # replace negative values with 0
    smoothed_block = smoothed_block.where(smoothed_block >= 0, 0)

    LOG.info(f'smoothn {window_size} successful')
    smoothed_block.to_netcdf(save_path)
    LOG.info(f'{save_path} successfully saved')

def process_large_dask_chunks(dask_dataset, block_size, var_name, flag, output_dir):
    """
    Process a large Dask dataset by splitting it into smaller spatial blocks, smoothing each block,
    saving the results, and then recombining them.
    """
    latitude_chunks = range(0, dask_dataset.latitude.size, block_size)
    longitude_chunks = range(0, dask_dataset.longitude.size, block_size)

    # store paths to all blocks after smoothn1 + linear + smoothn2
    processed_blocks = []

    os.makedirs(output_dir, exist_ok=True)

    with ProgressBar():
        for lat_start in latitude_chunks:
            for lon_start in longitude_chunks:
                lat_end = min(lat_start + block_size, dask_dataset.latitude.size)
                lon_end = min(lon_start + block_size, dask_dataset.longitude.size)

                block = dask_dataset.isel(latitude=slice(lat_start, lat_end),
                                          longitude=slice(lon_start, lon_end))

                # interpolation
                block = interpolate_block(block)

                # 1st smoother
                window_size1 = 1.5
                block_path = f"{output_dir}/smoothed_block_{lat_start}_{lon_start}.linear.smoothn{window_size1}.nc"
                process_and_save_block(block, block_path, var_name, window_size1, flag)

                processed_blocks.append(block_path)

    logging.info('Opening all smoothed blocks in one xarray')
    processed_datasets = xr.open_mfdataset(processed_blocks, chunks={"latitude": 256, "longitude": 256})

    logging.info('Smoothed blocks successfully combined')
    return processed_datasets


def process_large_dask_chunks_2(dask_dataset, block_size, var_name, period, output_dir):
    """
    Process a large Dask dataset by splitting it into smaller spatial blocks, smoothing each block,
    saving the results, and then recombining them.
    """
    latitude_chunks = range(0, dask_dataset.latitude.size, block_size)
    longitude_chunks = range(0, dask_dataset.longitude.size, block_size)

    # store paths to all blocks after smoothn1 + linear + smoothn2
    processed_blocks = []

    os.makedirs(output_dir, exist_ok=True)

    with ProgressBar():
        for lat_start in latitude_chunks:
            for lon_start in longitude_chunks:
                lat_end = min(lat_start + block_size, dask_dataset.latitude.size)
                lon_end = min(lon_start + block_size, dask_dataset.longitude.size)

                block = dask_dataset.isel(latitude=slice(lat_start, lat_end),
                                          longitude=slice(lon_start, lon_end))

                # detrending
                save_path = f"{output_dir}/{lat_start}_{lon_start}.dtr.{period}.nc"

                result = block.rolling(time=period, min_periods=1, center=True).mean()
                result.to_netcdf(save_path)
                # result = result.to_dataset(name=var_name)

                processed_blocks.append(save_path)
                LOG.info(f'{save_path} successfully saved')

    logging.info('Opening all detrended blocks in one xarray')
    processed_datasets = xr.open_mfdataset(processed_blocks, chunks={"latitude": 256, "longitude": 256})

    logging.info('Detrending blocks successfully combined')
    return processed_datasets


def main(site_dir):
    LOG.info(f"Processing site directory: {site_dir}")

    mleonn_path = os.path.join(site_dir, 'Sentinel', 'MSIL2A', 'datacube', 'MLEONN', 'lai')
    paths = glob(mleonn_path)

    # If paths list is empty
    if not paths:
        raise FileNotFoundError(f"No MLEONN data products folders found in the specified directory path {mleonn_path}.")

    # Loop over the different MLEONN data products i.e. LAI, cab ...
    for p in paths:
        var_name = os.path.basename(p)
        LOG.info(f'Starting interpolation for {var_name}')

        # get the tif files to interpolate
        tif_files = sorted(glob(os.path.join(p, "*.tif")))
        # Load and stack the files
        dask_chunks = concat_xr(tif_files, var_name)
        # create directory to interpol
        interp_dir = create_dir(p, 'interpolated')

        LOG.info('Begin block processing')
        output_blocks_dir = create_dir(interp_dir, 'dask_processing_outputs')
        processed_result = process_large_dask_chunks(
            dask_dataset=dask_chunks,
            block_size=10,  # Spatial block size
            var_name=var_name,  # Variable to smooth
            flag=True,  # Smoothing flag
            output_dir=output_blocks_dir,
        )

        LOG.info('The dask chunks have been processed')

        fname = f'{var_name}.linear.smoothn1.5.tif'
        output_dir = os.path.join(interp_dir, fname)

        # set extract and set proj 4 to xarray
        get_proj4_from_tif(tif_files[0], xarray=processed_result)
        save_xarray_old(output_dir, processed_result, var_name)

        # Detrend with a period of 73, check how many observations do we have per year
        dask_chunks = processed_result.chunk({"time": -1, "latitude": 10, "longitude": 10})
        period = 182
        output_blocks_dir = create_dir(interp_dir, 'dask_processing_outputs_dtr')
        ds_dtr = process_large_dask_chunks_2(
            dask_dataset=dask_chunks,
            block_size=10,  # Spatial block size
            var_name=var_name,  # Variable to smooth
            period=period,  # Smoothing flag
            output_dir=output_blocks_dir,
        )

        LOG.info(f'Detrending MLEONN product')
        ds_dtr = ds_dtr.astype('float32')
        get_proj4_from_tif(tif_files[0], xarray=ds_dtr)

        fname = f'{var_name}.linear.smoothn1.5.detrended.{period}.tif'
        output_dir = os.path.join(interp_dir, fname)
        save_xarray_old(output_dir, ds_dtr, var_name)

        LOG.info('Script process Done!')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <site_root_data_dir>")
    else:
        site_directory = sys.argv[1]
        main(site_directory)
