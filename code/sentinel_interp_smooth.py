import sys
import os
import xarray as xr
import logging
from glob import glob
from dask.diagnostics import ProgressBar
from WorldPeatland.code.utils import create_dir, proj4_extract
from WorldPeatland.code.gdal_sheep import gdal_stack_dt, create_xarr
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from WorldPeatland.code.smoothn import smoothn
from WorldPeatland.code.s1_pre_processing import process_and_save_block


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def process_large_dask_chunks(dask_dataset, block_size, var_name, window_size, flag, output_dir):
    """
    Process a large Dask dataset by splitting it into smaller spatial blocks, smoothing each block,
    saving the results, and then recombining them.
    """
    latitude_chunks = range(0, dask_dataset.latitude.size, block_size)
    longitude_chunks = range(0, dask_dataset.longitude.size, block_size)
    smoothed_blocks = []

    os.makedirs(output_dir, exist_ok=True)

    with ProgressBar():
        for lat_start in latitude_chunks:
            for lon_start in longitude_chunks:
                lat_end = min(lat_start + block_size, dask_dataset.latitude.size)
                lon_end = min(lon_start + block_size, dask_dataset.longitude.size)

                block = dask_dataset.isel(latitude=slice(lat_start, lat_end),
                                          longitude=slice(lon_start, lon_end))
                block_path = f"{output_dir}/smoothed_block_{lat_start}_{lon_start}.nc"
                process_and_save_block(block, block_path, var_name, window_size, flag)
                logging.info(f'{block_path} successfully saved')
                smoothed_blocks.append(block_path)

    logging.info('Opening all smoothed blocks in one xarray')
    smoothed_datasets = xr.open_mfdataset(smoothed_blocks, chunks={"latitude": 256, "longitude": 256})

    # Clean up temporary output blocks
    # for block_path in smoothed_blocks:
    #     os.remove(block_path)
    # os.rmdir(output_dir)
    # logging.info('Smoothed blocks successfully combined and temporary files deleted')
    return smoothed_datasets


def main(site_dir):
    LOG.info(f"Processing site directory: {site_dir}")

    mleonn_path = os.path.join(site_dir, 'Sentinel', 'MSIL2A', 'datacube', 'MLEONN', '*')
    paths = glob(mleonn_path)

    if not paths:  # If paths list is empty
        raise FileNotFoundError(f"No MLEONN data products folders found in the specified directory path {mleonn_path}.")

    # Loop over the different MLEONN data products i.e. LAI, cab ...
    for p in paths:
        # TODO remove this line later
        p = '/wp_data/sites/Degero/Sentinel/MSIL2A/datacube/MLEONN/lai'
        var_name = os.path.basename(p)
        LOG.info(f'Starting interpolation for {var_name}')

        # get the tif files to interpolate
        tif_files = sorted(glob(os.path.join(p, "*.tif")))
        #TODO remove this line later
        tif_files = tif_files[:5]
        # Load and stack the files
        stacked_arr, dts, saved_opn = gdal_stack_dt(tif_files)
        data_with_nan = create_xarr(saved_opn, var_name, stacked_arr, dts)
        # create directory to interpol
        interp_dir = create_dir(p, 'interpolated')
        proj4_string = proj4_extract(saved_opn)

        # 1. smoothn with a small window
        s = 3
        LOG.info(f'Starting first smoother with smoothing factor {s} for {var_name}')
        dask_chunks = data_with_nan.chunk({"latitude": 5, "longitude": 5})

        LOG.info('Begin block smoothing')
        output_blocks_dir = create_dir(interp_dir, 'output_blocks_smooth1')
        smoothed_result = process_large_dask_chunks(
            dask_dataset=dask_chunks,
            block_size=100,  # Spatial block size
            var_name=var_name,  # Variable to smooth
            window_size=s,  # Smoothing window size
            flag=True,  # Smoothing flag
            output_dir=output_blocks_dir,
        )

        LOG.info('The dask chunks have been smoothened')

        fname = f'{var_name}.smoothn.{s}.tif'
        output_dir = os.path.join(interp_dir, fname)
        # set extract and set proj 4 to xarray
        smoothed_result.attrs['crs'] = proj4_string
        save_xarray_old(output_dir, smoothed_result, var_name)

        LOG.info(f"First Smoothn  with smoothing factor of {s} successfully saved here {output_dir}")

        # 2. Interpolate missing values
        data_interpolated = smoothed_result.interpolate_na(dim='time', method='linear')
        fname = f'{var_name}.smoothn.{s}.linear.tif'
        output_dir = os.path.join(interp_dir, fname)
        # set proj 4 to xarray
        data_interpolated.attrs['crs'] = proj4_string
        save_xarray_old(output_dir, data_interpolated, var_name)

        # 3. apply smoothn on the interpolated data
        s = 30
        LOG.info(f'Starting smoothn 2 with {s}for {var_name}')

        LOG.info(f'Starting first smoother with smoothing factor {s} for {var_name}')
        dask_chunks = data_interpolated.chunk({"latitude": 5, "longitude": 5})

        LOG.info('Begin block smoothing')
        smoothed_result = process_large_dask_chunks(
            dask_dataset=dask_chunks,
            block_size=100,  # Spatial block size
            var_name=var_name,  # Variable to smooth
            window_size=s,  # Smoothing window size
            flag=True,  # Smoothing flag
            output_dir=output_blocks_dir,
        )

        LOG.info('The dask chunks have been smoothened')

        fname = f'{var_name}.linear.smoothn.0.5.tif'
        output_dir = os.path.join(interp_dir, fname)
        # set extract and set proj 4 to xarray
        smoothed_result.attrs['crs'] = proj4_string
        save_xarray_old(output_dir, smoothed_result, var_name)

        LOG.info(f'Smoothn 2 successfully saved here {output_dir}')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <site_root_data_dir>")
    else:
        site_directory = sys.argv[1]
        main(site_directory)
