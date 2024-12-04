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
    create_xr function takes as input asc or desc and returns an xr.Dataset with 3 data variables VV, VH, and angles
    of the input orbit.
    """
    bands = []
    stack_arr = []
    for i, j in fdict.items():
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
        sorted(glob.glob(f'{site_fpath}/%s/*/*.tif' % band)) for band in bands
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


def dask_smooth(in_ds, var2smooth, window, isrobust):
    """
    Smooth a given variable in a dataset using smoothn.

    INPUTS:
        - in_ds: Input xarray Dataset
        - var2smooth: Variable to smooth
        - window: Smoothing window parameter (s in smoothn)
        - isrobust: Whether to use robust smoothing

    OUTPUTS:
        - our_xr_ds: Smoothed xarray Dataset
    """
    smooth_ar = smoothn(y=in_ds[var2smooth].values,
                        s=window, isrobust=isrobust, axis=0)[0]

    our_xr_ds = xr.Dataset({var2smooth: (('time', 'latitude', 'longitude'),
                                         smooth_ar)},
                           coords=in_ds.coords)
    return our_xr_ds


def main(site_fpath):

    orbits = ['ASCENDING', 'DESCENDING']

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
            _ds = apply_threshold(ds)

            # TODO separate per angles??

            ds_cr = calc_cr(_ds)

            # TODO think about if 2 zones for one site
            # then will have to get the zone that is covering the largest area
            proj4_string = get_proj4_from_tif(file, xarray=ds_cr)

            output_utm = output_dir + f'/cross_ratio_{orbit}_utm_{timestep}.tif'
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
        ds_all_years = create_xarr(saved_opn, 'cr', stacked_arr, dts)

        # smoothn should happen on all the time series per orbit
        # Chunk the dataset to enable Dask computation
        dask_chunks = ds_all_years.chunk({"latitude": 5, "longitude": 5})

        # Apply the smoothing function using map_blocks
        # dask_output is a xarray.Dataset
        with ProgressBar():
            dask_output = xr.map_blocks(
                dask_smooth,
                dask_chunks,
                args=('cr', 3, True)  # Pass variable name, window size, and robust flag
            ).compute()

        dask_output.attrs['crs'] = proj4_string

        output_ts_dir = create_dir(output_dir, 'timeSeries')
        fname = output_ts_dir + f'/cross_ratio_{orbit}_utm_smoothn.tif'
        save_xarray_old(fname, dask_output, f'cr')
        LOG.info(f'Cross Ratio {orbit} Smoothened and saved here: {fname}')


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <site_fpath>")
    else:
        site_directory = sys.argv[1]  # i.e. '/wp_data/sites/Degero/Sentinel/datacube/S1_GRD'
        main(site_directory)
