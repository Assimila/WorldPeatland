import sys
import os
import xarray as xr
import logging
from glob import glob
from dask.diagnostics import ProgressBar
from WorldPeatland.code.utils import create_dir, proj4_extract
from WorldPeatland.code.gdal_sheep import gdal_stack_dt, create_xarr
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from WorldPeatland.code.s1_pre_processing import process_and_save_block

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def interpolate_and_save_block(block, save_path):
    """
    interpolate a single block and save it to disk.
    """
    interpolated_result = block.interpolate_na(dim='time', method='linear')
    interpolated_result.to_netcdf(save_path)
    LOG.info(f'{save_path} successfully saved')


def process_large_dask_chunks(dask_dataset, block_size, var_name , flag, output_dir):
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

                # 1st smoother
                window_size = 3
                block_path_s1 = f"{output_dir}/smoothed_block_{lat_start}_{lon_start}.smoothn{window_size}.nc"
                process_and_save_block(block, block_path_s1, var_name, window_size, flag)
                logging.info(f'{block_path_s1} successfully saved')


                # interpolation
                block_path_s1_linear = block_path_s1.replace('.nc', '.linear.nc')
                interpolate_and_save_block(block, block_path_s1_linear)


                # 3rd smoother
                window_size = 30
                block_path_s1_linear_s2 = block_path_s1_linear.replace('.nc', f'.smoothn{window_size}.nc')
                process_and_save_block(block, block_path_s1_linear_s2, var_name, window_size, flag)

                processed_blocks.append(block_path_s1_linear_s2)


    logging.info('Opening all smoothed blocks in one xarray')
    processed_datasets = xr.open_mfdataset(processed_blocks, chunks={"latitude": 256, "longitude": 256})

    # Clean up temporary output blocks
    # for block_path in smoothed_blocks:
    #     os.remove(block_path)
    # os.rmdir(output_dir)
    # logging.info('Smoothed blocks successfully combined and temporary files deleted')
    return processed_datasets


def main(site_dir):
    LOG.info(f"Processing site directory: {site_dir}")

    mleonn_path = os.path.join(site_dir, 'Sentinel', 'MSIL2A', 'datacube', 'MLEONN', '*')
    paths = glob(mleonn_path)


    if not paths:  # If paths list is empty
        raise FileNotFoundError(f"No MLEONN data products folders found in the specified directory path {mleonn_path}.")

    # Loop over the different MLEONN data products i.e. LAI, cab ...
    for p in paths:
        p = '/wp_data/sites/Degero/Sentinel/MSIL2A/datacube/MLEONN/lai'
        var_name = os.path.basename(p)
        LOG.info(f'Starting interpolation for {var_name}')

        # get the tif files to interpolate
        tif_files = sorted(glob(os.path.join(p, "*.tif")))
        # Load and stack the files
        stacked_arr, dts, saved_opn = gdal_stack_dt(tif_files)
        data_with_nan = create_xarr(saved_opn, var_name, stacked_arr, dts)
        # create directory to interpol
        interp_dir = create_dir(p, 'interpolated_test')
        proj4_string = proj4_extract(saved_opn)

        LOG.info('Dask chucking began')
        dask_chunks = data_with_nan.chunk({"latitude": 5, "longitude": 5})

        LOG.info('Begin block processing]')
        output_blocks_dir = create_dir(interp_dir, 'dask_processing_outputs')
        processed_result = process_large_dask_chunks(
            dask_dataset=dask_chunks,
            block_size=100,  # Spatial block size
            var_name=var_name,  # Variable to smooth
            flag=True,  # Smoothing flag
            output_dir=output_blocks_dir,
        )

        LOG.info('The dask chunks have been processed')

        fname = f'{var_name}.smoothn3.linear.smoothn30.tif'
        output_dir = os.path.join(interp_dir, fname)

        # TODO remove tests
        # smoothed_result = rioxarray.open_rasterio(output_dir)
        # data = smoothed_result.values
        # smoothed_result = smoothed_result.rename({'x': 'lon', 'y': 'lat', 'band': 'time'})
        # smoothed_result = smoothed_result.assign_coords({
        #     'time': dts,  # Replace band indices with datetime values
        # })

        # set extract and set proj 4 to xarray
        processed_result.attrs['crs'] = proj4_string
        save_xarray_old(output_dir, processed_result, var_name)
        LOG.info('Script process Done!')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <site_root_data_dir>")
    else:
        site_directory = sys.argv[1]
        main(site_directory)
