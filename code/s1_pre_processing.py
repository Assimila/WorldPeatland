from glob import glob
import numpy as np
import xarray as xr
from osgeo import gdal
import os
from dask.diagnostics import ProgressBar
from itertools import chain
from collections import defaultdict
import logging
import sys
import matplotlib.pyplot as plt
import pandas as pd
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from WorldPeatland.code.gdal_sheep import gdal_dt, create_coord_list, create_xarr, _get_times, get_proj4_from_tif, _get_FillValue
from WorldPeatland.code.utils import create_dir, get_timestep_from_tif
from WorldPeatland.code.smoothn import smoothn


sys.path.insert(0, '/workspace/WorldPeatland/code/')

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def create_xr(fdict):
    """
    create_xr function takes as input asc or desc and returns a xr.Dataset with 3 data variables VV, VH, and angles
    of the input orbit.
    """
    bands = []
    stack_arr = []
    for i, j in fdict.items():

        LOG.info(f'Opening tif file in xarray: {j}')
        arr, dts, opn = gdal_dt(j)
        stack_arr.append(arr)
        bands.append(i)
    stack_dict = dict(zip(bands, stack_arr))

    # Check if there is only one timestep
    if len(dts) == 1:
        # Add a new axis to each array in stack_dict to include the time dimension
        for bd in stack_dict:
            stack_dict[bd] = stack_dict[bd][np.newaxis, :, :]  # Reshape to (1, latitude, longitude)

    xs, ys = create_coord_list(opn)
    ds = xr.Dataset(
        data_vars={bd: (('time', 'latitude', 'longitude'), stack_dict[bd]) for bd in bands},
        coords={'time': dts,
                'latitude': ys,
                'longitude': xs}
    )

    return ds


def apply_threshold(ds):
    for var_name in ds.data_vars:
        if var_name.startswith('V'):
            ds[var_name] = ds[var_name].where(ds[var_name] > -30, 999)  # _FillValue = 999 of Sentinel 1
    return ds


def calc_cr(ds):
    """
    calc_cr function takes as input the xarray.Dataset, returns a new xarray with cross ratio calculated,
    and saves it as a netcdf file. 
    
    INPUTS:
        - orbit (string) - ASCENDING or DESCENDING

    OUTPUTS:
        - ds (xarray) - with cross ratio as the 4th variable (in total 4 variables)

    """

    # Calculate Cross Ratio 
    ds = ds.assign(cr=ds[f"VH"] - ds[f'VV'])

    return ds


def group_tifs_by_date(site_fpath, orbit):
    """
    Groups `.tif` files by their timestep (date) based on the provided directory and orbit.

    Args:
        site_fpath (str): Path to the base directory containing `.tif` files.
        orbit (str): The orbit name to filter the bands.

    Returns:
        dict: A dictionary where keys are timesteps (dates) and values are lists of `.tif` file paths.
    """
    # Define the required bands
    bands = [f'VV_{orbit}', f'VH_{orbit}', f'angle_{orbit}']

    # Get a flattened list of all `.tif` file paths for the corresponding bands
    flat_list = list(chain.from_iterable(
        sorted(glob(f'{site_fpath}/%s/*/*.tif' % band)) for band in bands
    ))

    # Initialize a default dict to hold lists of file paths for each date
    grouped_tifs = defaultdict(list)

    # Iterate through each `.tif` file
    for month_tif in flat_list:
        # Extract the timestep (date) from the filename
        timestep = get_timestep_from_tif(month_tif)

        # Group the file under its corresponding timestep
        grouped_tifs[timestep].append(month_tif)

    # Convert default dict to a regular dictionary and return it
    return dict(grouped_tifs)


def dw_smoothn_smooth_xarray(in_ds, var2smooth, window, isrobust):
    """
    Smooth a given variable in a dataset using smoothn.
    source: Alex

    INPUTS:
        - in_ds (xarray.Dataset): Input xarray Dataset.
        - var2smooth (str): Name of the variable to smooth.
        - window (float): Smoothing window parameter (s in smoothn).
        - isrobust (bool): Whether to use robust smoothing.

    OUTPUTS:
        - xarray.Dataset: A dataset containing the smoothed variable.
    """

    # Handle empty or all-NaN cases
    if in_ds[var2smooth].size == 0:
        return in_ds
    if np.all(np.isnan(in_ds[var2smooth].values)):
        return xr.Dataset(
            data_vars={
                var2smooth: (('time', 'latitude', 'longitude'), in_ds[var2smooth].values)
            },
            coords=in_ds.coords,
        )

    # Apply smoothn
    smooth_ar = smoothn(y=in_ds[var2smooth].values, s=window, isrobust=isrobust, axis=0)[0]

    # Convert to float32 before returning
    smooth_ar = smooth_ar.astype(np.float32)

    # Return the smoothed dataset
    return xr.Dataset(
        data_vars={
            var2smooth: (('time', 'latitude', 'longitude'), smooth_ar)
        },
        coords=in_ds.coords,
    )


def process_and_save_block(block, save_path, var_name, window_size, flag):
    """
    Smooth a single block and save it to disk.
    """
    smoothed_block = dw_smoothn_smooth_xarray(block, var_name, window_size, flag)
    smoothed_block.to_netcdf(save_path)


def process_large_dask_chunks(dask_dataset, block_size, var_name, window_size, flag, output_dir):
    """
    Process a large Dask dataset by splitting it into smaller spatial blocks, smoothing each block,
    saving the results, and then recombining them.
    """
    latitude_chunks = range(0, dask_dataset.latitude.size, block_size)
    longitude_chunks = range(0, dask_dataset.longitude.size, block_size)

    smoothed_blocks = []
    # smoothed_blocks=glob(f'{output_dir}/*.nc')
    with ProgressBar():
        for lat_start in latitude_chunks:
            for lon_start in longitude_chunks:
                # Define the slice for the current block
                lat_end = min(lat_start + block_size, dask_dataset.latitude.size)
                lon_end = min(lon_start + block_size, dask_dataset.longitude.size)

                # Select the block
                block = dask_dataset.isel(latitude=slice(lat_start, lat_end),
                                          longitude=slice(lon_start, lon_end))

                # Smooth the block and save it
                block_path = f"{output_dir}/smoothed_block_{lat_start}_{lon_start}.nc"
                process_and_save_block(block, block_path, var_name, window_size, flag)
                LOG.info(f'{block_path} successfully saved')
                smoothed_blocks.append(block_path)

    LOG.info('Finish processing all blocks')
    # Recombine the saved blocks
    LOG.info('Opening all smoothed blocks nc in one xarray')
    smoothed_datasets = xr.open_mfdataset(smoothed_blocks, chunks={"latitude": 256, "longitude": 256})
    LOG.info('Smoothened blocks successfully combined')
    return smoothed_datasets  # xarr dataset


def main(site_fpath):

    orbits = ['ASCENDING','DESCENDING']  #

    for orbit in orbits:

        orbit_files = [file for file in os.listdir(site_fpath) if orbit in file]
        if not orbit_files:
            LOG.info(f"No files found with orbit '{orbit}' in {site_fpath}. Skipping to the next orbit.")
            continue  # Skip to the next orbit if no files are found

        output_dir = create_dir(site_fpath, f'CrossRatio_{orbit}')

        grouped_tifs = group_tifs_by_date(site_fpath, orbit)

        monthly_outputs = []

        for timestep, files in grouped_tifs.items():

            # Create the dictionary from files to stack and concatenate the datasets in xarray format
            files_dict = {}
            for file in files:
                if "VV" in file:
                    files_dict["VV"] = file
                elif "VH" in file:
                    files_dict["VH"] = file
                elif "angle" in file:
                    files_dict["angle"] = file

            ds = create_xr(files_dict)
            # Threshold for backscatter values not angular values
            _ds = apply_threshold(ds)

            # get the angular information distribution
            a = ds.angle.values
            LOG.info(f'Check with that the histogram printed is following a binomial distribution')
            LOG.info(f'Check that this mean value of {np.nanmean(a)} is in the middle of the angles ')
            plt.hist(a.flatten(), bins=30, edgecolor='black')
            plt.show()
            angular_threshold = np.nanmean(a)

            # Create a condition where angle is less than angular_threshold
            condition = _ds.angle < angular_threshold
            # Use where to drop values based on the condition (drop NaNs where condition is false)
            ds_o1 = _ds.where(condition, drop=True)

            # Create a condition
            condition = _ds.angle > angular_threshold
            # Use where to drop values based on the condition (drop NaNs where condition is false)
            ds_o2 = _ds.where(condition, drop=True)

            for ds_subset, angle_label in [(ds_o1, 'angle1'), (ds_o2, 'angle2')]:
                LOG.info(f'calculating cross ratio for {timestep} - {orbit} orbit - {angle_label}')

                ds_cr = calc_cr(ds_subset)
                LOG.info(f'{timestep} Cross ratio {orbit} for {angle_label} successfully calculated')

                # TODO: Handle cases where one site has two zones
                proj4_string = get_proj4_from_tif(file, xarray=ds_cr)

                opn = gdal.Open(file)
                fill_value = _get_FillValue(opn)
                ds_cr.attrs['fill_value'] = fill_value

                output_dir_ang = create_dir(output_dir, angle_label)
                output_utm = f'{output_dir_ang}/cross_ratio_{orbit}_utm_{timestep}.tif'

                monthly_outputs.append(output_utm)
                save_xarray_old(output_utm, ds_cr, 'cr')

            # # change projection from utm to sinusoidal
            #       output_sinu = transform_save(ds_cr, orbit, saved_path)
            #       LOG.info(f'Cross Ratio for {orbit} has been saved here {saved_path}')
            #
            #       # Create a xarray to be able to perform smoothn
            #       arr, dts, saved_opn = gdal_dt(output_sinu, 'time')
            #       ds = create_xarr(saved_opn, 'cr', arr, dts)

        # open all cross ratio tiffs and put in one xarray to be able to smoothn
        # stacked_arr, dts, saved_opn = gdal_stack_dt(monthly_outputs)
        list_xr = []
        for fpath in monthly_outputs:
            ds = xr.open_dataset(fpath, chunks={"latitude": 256, "longitude": 256})
            # Rename dimensions
            ds = ds.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time', 'band_data': 'cr'})
            times = _get_times(fpath)
            ds['time'] = times.time.values
            list_xr.append(ds)
        ds_all_years = xr.concat(list_xr, dim='time')

        LOG.info('Open all Cross ratio tiffs in one xr before dask processing')

        # smoothn should happen on all the time series per orbit
        # Chunk the dataset to enable Dask computation
        LOG.info('Begin dask chunking')
        dask_chunks = ds_all_years.chunk({"latitude": 5, "longitude": 5})

        LOG.info('Begin block smoothing')
        output_blocks_dir = create_dir(output_dir, 'output_blocks')
        smoothed_result = process_large_dask_chunks(
            dask_dataset=dask_chunks,
            block_size=100,  # Spatial block size
            var_name="cr",  # Variable to smooth
            window_size=3,  # Smoothing window size
            flag=True,  # Smoothing flag
            output_dir=output_blocks_dir,
        )

        del ds_all_years

        LOG.info('The dask chunks have been smoothened')
        smoothed_result.attrs['crs'] = proj4_string
        smoothed_result.attrs['fill_value'] = fill_value
        output_ts_dir = create_dir(output_dir, 'timeSeries')
        fname = output_ts_dir + f'/cross_ratio_{orbit}_utm_smoothn.tif'
        save_xarray_old(fname, smoothed_result, f'cr')
        LOG.info(f'Cross Ratio {orbit} Smoothened and saved here: {fname}')


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <site_fpath>")
    else:
        site_directory = sys.argv[1]  # i.e. '/wp_data/sites/Degero/Sentinel/datacube/S1_GRD'
        main(site_directory)
