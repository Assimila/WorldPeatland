from WorldPeatland.code.save_xarray_to_gtiff_old import *
from WorldPeatland.code.gdal_sheep import *
from WorldPeatland.code.MLEO_NN import *
from WorldPeatland.code.utils import *
from smoothn import smoothn
from dask.diagnostics import ProgressBar
from itertools import chain
from collections import defaultdict
import logging
import sys

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
            ds[var_name] = ds[var_name].where(ds[var_name] > -30, np.nan)
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


def transform_save(orbit, saved_path):
    # save_xarray can only save one variable
    # do we create a new one that can fit more than one variable?
    # because need to save observation and cross ratio 
    # ==>> maybe try to save xarray with many variables for each orbit.
    # 'ex: ascending orbit has a xarray with cross ratio, VV and VH'
    output_utm = saved_path + f'/cross_ratio_{orbit}_utm.tif'
    #     save_xarray_old(output_utm, ds, 'cr')

    # change projection from utm to sinusoidal 
    output_sinu = saved_path + f'/cross_ratio_{orbit}_sinusoidal_resampled.tif'
    proj4_string = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs '
    ds = gdal.Open(output_utm)

    # reproject to sinusoidal and resample to 10 by 10 pixel size
    gdal.Warp(output_sinu, ds, dstSRS=proj4_string, xRes=10, yRes=10)

    # delete utm files
    os.remove(output_utm)

    return output_sinu


def group_tifs_by_date(site_fpath, orbit, orbit_no):
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
        sorted(glob.glob(f'{site_fpath}/%s/*/*{orbit_no}*.tif' % band)) for band in bands
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


def DW_smoothn_smooth_xarray(in_ds, var2smooth, window, isrobust):
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
    print(in_ds[var2smooth].values.shape)

    # Retrieve dimensions
    # lat_size = in_ds.dims['latitude']
    # lon_size = in_ds.dims['longitude']

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
    smoothed_block = DW_smoothn_smooth_xarray(block, var_name, window_size, flag)
    smoothed_block.to_netcdf(save_path)


def process_large_dask_chunks(dask_dataset, block_size, var_name, window_size, flag, output_dir):
    """
    Process a large Dask dataset by splitting it into smaller spatial blocks, smoothing each block,
    saving the results, and then recombining them.
    """
    latitude_chunks = range(0, dask_dataset.latitude.size, block_size)
    longitude_chunks = range(0, dask_dataset.longitude.size, block_size)

    smoothed_blocks = []

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
    smoothed_datasets = [xr.open_dataset(path) for path in smoothed_blocks]
    combined = xr.combine_by_coords(smoothed_datasets)
    LOG.info('Smoothened blocks successfully combined')
    return combined


def main(site_fpath, orbit, orbit_no):

    output_dir = create_dir(site_fpath, f'CrossRatio_{orbit}')

    grouped_tifs = group_tifs_by_date(site_fpath, orbit, orbit_no)

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
        _ds = apply_threshold(ds)

        LOG.info(f'calculating cross ratio for {timestep} - {orbit} orbit')
        ds_cr = calc_cr(_ds)
        LOG.info(f'{timestep} Cross ratio {orbit} successfully calculated')

        # TODO think about if 2 zones for one site
        # then will have to get the zone that is covering the largest area
        proj4_string = get_proj4_from_tif(file, xarray=ds_cr)

        output_utm = output_dir + f'/cross_ratio_{orbit}_{orbit_no}_utm_{timestep}.tif'
        monthly_outputs.append(output_utm)
        save_xarray_old(output_utm, ds_cr, 'cr')

            # change projection from utm to sinusoidal
        #           output_sinu = transform_save(ds_cr, orbit, saved_path)
        #           LOG.info(f'Cross Ratio for {orbit} has been saved here {saved_path}')

        #           # Create an xarray to be able to perform smoothn
        #           arr, dts, saved_opn = gdal_dt(output_sinu, 'time')
        #           ds = create_xarr(saved_opn, 'cr', arr, dts)
        # open all cross ratio tiffs and put in one xarray
    stacked_arr, dts, saved_opn = gdal_stack_dt(monthly_outputs)

    LOG.info('Open all Cross ratio tiffs in one xr before dask processing')
    ds_all_years = create_xarr(saved_opn, 'cr', stacked_arr, dts)

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
        output_dir=output_blocks_dir
    )

    smoothed_result.to_netcdf("smoothed_dataset.nc")
    LOG.info('The dask chunks have been smoothened')
    smoothed_result.attrs['crs'] = proj4_string

    output_ts_dir = create_dir(output_dir, 'timeSeries')
    fname = output_ts_dir + f'/cross_ratio_{orbit}_{orbit_no}_utm_smoothn.tif'
    save_xarray_old(fname, smoothed_result, f'cr')
    LOG.info(f'Cross Ratio {orbit} {orbit_no} Smoothened and saved here: {fname}')


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python script.py <site_fpath_S1_GRD> <orbit> <orbit_no>")
        print('Example: python s1_pre_processing /path/to/site/S1_GRD ASCENDING 58')

    else:
        site_directory = sys.argv[1]  # i.e. '/wp_data/sites/Degero/Sentinel/datacube/S1_GRD'
        orbit = sys.argv[2].upper()
        orbit_no = sys.argv[3]

        if not orbit_no.isdigit():
            print("Error: orbit_no must be a whole number like 58")
            sys.exit(1)

        orbit_no = int(orbit_no)  # Convert to integer after validation

        if orbit not in ['ASCENDING', 'DESCENDING']:
            print('Error: orbit direction must be <ASCENDING> or <DESCENDING>')
        main(site_directory, orbit, orbit_no)
