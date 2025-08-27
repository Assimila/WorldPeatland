import os
from osgeo import gdal, osr
from osgeo import gdal_array
import numpy as np
from datetime import datetime as dt
import xarray as xr
import gc
import subprocess
import re
import logging
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_NoDataValue(opn):
    """
    Get the Nodata value if set using SetNoDataValue
    """

    band = opn.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()

    return nodata_value


def _get_FillValue(opn):
    """
    Get _FillValue from band 1 (Randomly should all be the same)
    if set in written in band metadata
    """

    dst_band = opn.GetRasterBand(1)
    meta = dst_band.GetMetadata()
    x = None

    keys = ['_FillValue', 'fill_value']
    for key in keys:
        if key in meta:
            x = meta[key]
            break
    else:
        raise KeyError("No valid fill value key found in metadata.")

    return x


def gdal_dt(e):
    """
    gdal_dt function will open the tif file as an OSGeo gdal dataset

    INPUTS:
        - e (str or tiff) - path the tiff file or the gdal dataset you want to
            open and save its datetime
        - time (string) - check how the time variable is written in the tiff metadata
    Outputs:
        - arr (np.array) - return arr of the gdal dataset
        - dts (list) - list of the datetime
        - saved_opn (OSGeo gdal dataset) - saved dataset for its srs
    """

    # Create an empty list to store the datetime
    dts = []

    # Check if the input is a str which would be the tif file 
    # otherwise it is already an opened gdal dataset 
    if str == type(e):
        # open the Dataset
        opn = gdal.Open(e)
    else:
        opn = e

    for i in range(1, opn.RasterCount + 1):

        rst = opn.GetRasterBand(i)
        meta = rst.GetMetadata()

        # following fill in with the corresponding format 
        # 'time' check the metadata of the tiff to see what they call
        # could also be 'RANGEBEGINNINGDATE'
        # the time data
        time_keys = ['time', 'RANGEBEGINNINGDATE', 'DATE']
        for key in time_keys:
            if key in meta:
                x = meta[key]
                break
        else:
            raise KeyError("No valid time key found in metadata.")

        # Possible datetime formats to try
        dt_formats = [
            '%Y-%m-%dT%H:%M:%S.%f000',
            '%Y-%m-%dT%H:%M:%S.000000000',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d'
        ]

        # Initialize the datetime object
        t = None

        # Attempt to parse the datetime using the possible formats
        for dt_format in dt_formats:
            try:
                t = dt.strptime(x, dt_format)
                break  # Stop if parsing is successful
            except ValueError:
                continue  # Try the next format if parsing fails

        # Raise an error if all formats fail
        if t is None:
            raise ValueError(f"None of the datetime formats matched the value: {x}")

        # append it to the list
        dts.append(t)

    # save the last osegeodataset for its srs
    saved_opn = opn

    # open the array
    arr = opn.ReadAsArray()

    return arr, dts, saved_opn


def gdal_stack_dt(lt):
    """
    gdal_stack_dt function will open geo-tiffs files and concatenate the dataset,
    set the time as datetime (=> to open many geo-tiffs and stack them)

    INPUTS:
        - lt (list) - list containing all the geo-tiffs files to be concatenated

    OUTPUTS:
        - stacked_arr (np.array) - stacked array containing all the layers of the
            input
        - dts (list) - list of the datetime
        - saved_opn (OSGeo gdal dataset) - saved dataset for its srs
    """

    # Create empty lists
    ARRAYS_RESHAPED = []
    dts = []

    # loop through the files
    for e in lt:
        # Check if the input is a str which would be the tif file 
        # otherwise it is already an opened gdal dataset

        LOG.info(f'Currently loading {e}')

        if str == type(e):
            # open the Dataset
            opn = gdal.Open(e)
            if opn is None:
                raise FileNotFoundError(f'Unable to open file: {e}')
        else:
            opn = e

        # Gdal counts from 1 
        for i in range(1, opn.RasterCount + 1):

            rst = opn.GetRasterBand(i)
            meta = rst.GetMetadata()
            # following fill in with the corresponding format 
            # 'time' check the metadata of the tiff to see what they call
            # the time data            
            time_keys = ['time', 'RANGEBEGINNINGDATE', 'DATE']
            for key in time_keys:
                if key in meta:
                    x = meta[key]
                    break
            else:
                raise KeyError("No valid time key found in metadata.")

            # Possible datetime formats to try
            dt_formats = [
                '%Y-%m-%dT%H:%M:%S.000000000',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d'
            ]

            # Initialize the datetime object
            t = None

            # Attempt to parse the datetime using the possible formats
            for dt_format in dt_formats:
                try:
                    t = dt.strptime(x, dt_format)
                    break  # Stop if parsing is successful
                except ValueError:
                    continue  # Try the next format if parsing fails

            # Raise an error if all formats fail
            if t is None:
                raise ValueError(f"None of the datetime formats matched the value: {x}")

            # append it to the list
            dts.append(t)

        # save the last osegeodataset for its srs
        saved_opn = opn

        # open the array
        arr = opn.ReadAsArray()  # xoff=0, yoff = 0, xsize=100, ysize=100) # add these arguments to take only a chunk of
                                                                        # the dataset spatially

        # check the dimensions of the array because cannot concatenate
        # arrays with different dimensions they all should be 3d np.arrays
        # some bands will have a 2d arrays meaning they only have one image 
        # for one date and not many dates
        n = arr.ndim
        if n == 2:
            arr = arr[np.newaxis, :, :]

        # append it to the list
        ARRAYS_RESHAPED.append(arr)

    if len(ARRAYS_RESHAPED) > 0:
        stacked_arr = np.concatenate(ARRAYS_RESHAPED, axis=0)
    else:
        raise ValueError("No arrays to concatenate in ARRAYS_RESHAPED.")

    return (stacked_arr, dts,
            saved_opn)


def create_xarr(opn, var_name, arr, dts):
    """
    create_xarr function will create a xarray
    INPUTS:
        - opn (osgeo gdal dataset) -
        - var_name (string) - name of the variable, or the band or the information
            stored in the pixels of the tif, or in the array
        - arr (np.array) - contains all the data values to be stored from a tiff
            to a xarray
    OUTPUTS:
        - ds (xarray) - dataset of the xarray
    """
    # create the x and y list of coordinates 
    # GetGeotransform gets me the corner coordinates of the tiff
    # i.e. (564550.0, 10.0, 0.0, 5931390.0, 0.0, -10.0)
    params = opn.GetGeoTransform()  # params is a tuple 
    xs = np.array([params[0] + (params[1] * i) + (params[1] / 2) for i in np.arange(opn.RasterXSize)])
    # params[0] is the top left point x/lon coordinate value prams[1] is the length along the x-axis of 1 pixel
    # RasterXSize is the total number of pixels along the x-axis (later it would be the size of the whole tiff of
    # xarray)
    ys = np.array([params[3] + (params[5] * i) + (params[5] / 2) for i in np.arange(opn.RasterYSize)])
    # params[3] is the same point top left but now its y/lat coordinate value 
    # params[5] is the length or step to reach the second point along the y-axis of one pixel
    # it is - because you are going downward the y-axis or latitude line 

    variable_name = var_name

    # Check if arr is 2D make it 3D to fit the time dimension
    if arr.ndim == 2:
        arr = arr[np.newaxis, :]

    ds = xr.Dataset(data_vars={variable_name: (('time', 'latitude', 'longitude'), arr)},
                    coords={'time': dts,
                            'latitude': ys,
                            'longitude': xs})
    return ds


def create_coord_list(opn):
    """Path the osgeo gdal database that you need to get the references xs and ys"""
    params = opn.GetGeoTransform()
    xs = [params[0] + (params[1] * i) for i in np.arange(opn.RasterXSize)]
    ys = [params[3] + (params[5] * i) for i in np.arange(opn.RasterYSize)]

    # x = [params[0] + (params[1] * i) + (params[1] / 2) for i in np.arange(opn.RasterXSize)]
    # y = [params[3] + (params[5] * i) + (params[5] / 2) for i in np.arange(opn.RasterYSize)]

    return xs, ys


def reproject_image(source_img, target_img, clip_shapefile=None, no_data_val=-9999):
    """
    Taken from Alex
    Function to reproject a source image onto the exact same spatial grid, so it
    has the same extent and spatial resolution as the other. It first checks to see 
    if a reprojection is needed (as they can be slow) and then performs one. It 
    is also cut to a shapefile too if needed.
    INPUTS:
         - source_img (string) - path to the image you want to manipulate.
         - target_image (string) - path to what you want source image to 
              look like.
    OPTIONS:
        - clip_shapefile (string) - the path of the shapefile you want to clip
              the data to.
        - no_data_val (int/float) - no data value to use. This will be for the 
              data outside the shapefile.
    OUTPUTS:
        - a gdal dataset. to access the data use ReadAsArray()
    """

    # get the details of the source image
    if str == type(source_img):
        s = gdal.Open(source_img)
    else:
        s = target_img

    geo_s = s.GetGeoTransform()
    s_x_size, s_y_size = s.RasterXSize, s.RasterYSize
    s_xmin = min(geo_s[0], geo_s[0] + s_x_size * geo_s[1])
    s_xmax = max(geo_s[0], geo_s[0] + s_x_size * geo_s[1])
    s_ymin = min(geo_s[3], geo_s[3] + s_y_size * geo_s[5])
    s_ymax = max(geo_s[3], geo_s[3] + s_y_size * geo_s[5])
    s_xRes, s_yRes = abs(geo_s[1]), abs(geo_s[5])

    # get the details of the target image
    if type(target_img) == str:
        t = gdal.Open(target_img)
    else:
        t = target_img
    geo_t = t.GetGeoTransform()
    x_size, y_size = t.RasterXSize, t.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])

    if (s_x_size == x_size) & (s_y_size == y_size) & \
            (s_xmin == xmin) & (s_ymin == ymin) & \
            (s_xmax == xmax) & (s_ymax == ymax) & \
            (s_xRes == xRes) & (s_yRes == yRes):

        if clip_shapefile is not None:
            g = gdal.Warp('', source_img, format='MEM',
                          cutlineDSName=clip_shapefile,
                          cropToCutline=True, dstNodata=no_data_val)
        else:
            g = gdal.Open(source_img)

    else:

        dstSRS = osr.SpatialReference()
        raster_wkt = t.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)

        if clip_shapefile is not None:
            g = gdal.Warp('', source_img, format='MEM',
                          outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                          dstSRS=dstSRS, cutlineDSName=clip_shapefile,
                          cropToCutline=True, dstNodata=no_data_val)

        else:
            g = gdal.Warp('', source_img, format='MEM',
                          outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                          dstSRS=dstSRS)
    return g


def save_3d_masks(in_arr, gdalobj, save_name):
    """
    Alex's function

    INPUTS
        - in_arr (numpy array) - 3d array with the data to be saved in tif
        - gdalobj - of one image that has the same coordinate system & ...
        - save_name (string) - path with tif filename
    """
    dtype_non_gdal = in_arr.dtype
    if dtype_non_gdal == 'bool':
        dtype = gdal.GDT_Byte
    else:
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dtype_non_gdal)
    cols = in_arr.shape[2]
    rows = in_arr.shape[1]
    lyrcount = in_arr.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    driver_options = ['COMPRESS=DEFLATE',
                      'BIGTIFF=YES',
                      'PREDICTOR=1',
                      'TILED=YES',
                      'COPY_SRC_OVERVIEWS=YES']

    outRaster = driver.Create(save_name, cols, rows, lyrcount, dtype, driver_options)
    outRaster.SetGeoTransform(gdalobj.GetGeoTransform())
    for n, i in enumerate(in_arr):
        outband = outRaster.GetRasterBand(n + 1)
        outband.WriteArray(i)

    outRaster.SetProjection(gdalobj.GetProjection())
    outband.FlushCache()


def get_proj4_from_tif(tif_file, xarray=None):
    """
    Extracts the PROJ.4 string from a given GeoTIFF file and optionally assigns it to a xarray dataset.

    INPUTS:
        - tif_file (str): Path to the input GeoTIFF file.
        - xarray (xarray.Dataset, optional): An xarray dataset to which the CRS attribute will be added.

    OUTPUT:
        - proj4_string (str): The extracted PROJ.4 string.
        @rtype: str
    """
    try:
        # Command to extract PROJ.4 string using gdalinfo
        command = ['gdalinfo', tif_file, '-proj4']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors in the gdalinfo output
        if result.returncode != 0:
            raise RuntimeError(f"Error running gdalinfo: {result.stderr}")

        # Use regex to find the PROJ.4 string in the output
        proj4_match = re.search(r"PROJ\.4 string is:\n'(.*?)'", result.stdout)
        if not proj4_match:
            raise ValueError("PROJ.4 string not found in gdalinfo output.")

        proj4_string = proj4_match.group(1)

        # Optionally set the PROJ.4 string as an attribute in the xarray dataset
        if xarray is not None:
            xarray.attrs['crs'] = proj4_string

        return proj4_string

    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting the PROJ.4 string: {e}")


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


def load_one_ds(fpath, data_var):

    """Could run it like this datasets = [load_one_ds(fp, data_var) for fp in tif_fnames] """
    LOG.info(fpath)
    times = _get_times(fpath)

    ds = xr.open_dataset(fpath)
    ds = ds.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
    ds['time'] = times.time.values

    # rename to 'cr'
    ds = ds.rename({'band_data': data_var})
    LOG.info('Done opening tif into xr')

    return ds


def _get_times(tif_path):
    """
    Get date/time from per-band metadata for a single TIF file.
    """
    LOG.info(f'Extracting time metadata for {tif_path}')
    d = gdal.Open(tif_path)
    n_bands = d.RasterCount

    times = []

    # GDAL bands are 1-indexed
    for n_band in range(1, n_bands + 1):
        b = d.GetRasterBand(n_band)
        md = b.GetMetadata()

        # Check if 'time' metadata exists to avoid errors
        if 'time' in md:
            times.append(md['time'])
        else:
            # Handle cases where 'time' metadata is missing
            times.append(None)
    # Convert list to DataFrame
    times = pd.DataFrame(times, columns=['time'])
    # Change data type to np.datetime64
    times.time = pd.to_datetime(times['time'],
                                format='%Y-%m-%dT%H:%M:%S').to_numpy()
    return times


def preprocess_tif(da):
    """
    A preprocess function for xarray.open_mfdataset.
    This function will be applied to each dataset as it's opened.
    It adds a 'time' dimension and coordinate from the band metadata.
    """
    # Get the original file path from the dataset's encoding
    tif_path = da.encoding['source']
    LOG.info(f'Extracting metadata for {tif_path}')

    # Get the times for this specific file
    times = _get_times(tif_path)

    da = da.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
    LOG.info(f'Setting time metadata into xr for {tif_path}')
    da['time'] = times.time.values
    return da


def concat_xr(tif_fnames, data_var):
    """
    Concatenate multiple TIF files into a single xarray Dataset
    using the preprocess function.
    """

    LOG.info('Begin dask chunking for concat')
    ds = xr.open_mfdataset(
        tif_fnames,
        preprocess=preprocess_tif,
    )

    # Rename the data variable
    LOG.info(f'Set data variable name as {data_var}')
    ds = ds.rename({'band_data': data_var})

    ds_optimised = ds.chunk({"longitude": 10, "latitude": 10, "time": -1})

    LOG.info('Dask chunking successful for concat')

    del ds
    gc.collect()

    return ds_optimised


# ---- THE DIAGNOSTIC CHECK FUNCTION ----
def are_longitudes_equal(arr1, arr2):
    """
    Checks if the 'longitude' coordinates of two xarray DataArrays are equal.

    Args:
        arr1 (xr.DataArray): The first xarray DataArray.
        arr2 (xr.DataArray): The second xarray DataArray.

    Returns:
        bool: True if longitudes are equal, False otherwise.
    """
    # First, check if the longitude coordinate exists in both arrays
    if 'longitude' not in arr1.coords or 'longitude' not in arr2.coords:
        print("Error: 'longitude' coordinate not found in one or both arrays.")
        return False

    # Check if the number of values in the coordinates are the same
    if arr1.longitude.size != arr2.longitude.size:
        print("Longitude coordinates have different sizes.")
        print(f"Size of arr1.longitude: {arr1.longitude.size}")
        print(f"Size of arr2.longitude: {arr2.longitude.size}")
        return False

    # Finally, check if the coordinate values are equal.
    # We use .all() to ensure every single value is identical.
    if (arr1.longitude.values == arr2.longitude.values).all():
        print("Success: Longitude coordinates are identical.")
        return True
    else:
        print("Failure: Longitude coordinates are not identical.")
        print("Array 1 longitude:", arr1.longitude.values)
        print("Array 2 longitude:", arr2.longitude.values)
        return False


def is_time_monotonically_ordered_with_debug(datasets: List[xr.Dataset]) -> bool:
    """
    Checks if the time coordinate is monotonically increasing across a list of xarray Datasets.
    Prints the problematic time values if an error is found.

    Args:
        datasets (List[xr.Dataset]): A list of xarray Datasets to check.

    Returns:
        bool: True if the time coordinate is monotonically increasing, False otherwise.

    i.e.
    # Assuming 'datasets' is your list of xarray datasets
    is_time_monotonically_ordered_with_debug(datasets)

    """
    if not datasets:
        return True  # An empty list is considered ordered.

    # 1. Check if time exists in all datasets
    for i, ds in enumerate(datasets):
        if 'time' not in ds.coords:
            print(f"Error: 'time' coordinate not found in dataset at index {i}.")
            return False

    # 2. Check within each dataset
    for i, ds in enumerate(datasets):
        time_values = ds.coords['time'].values

        # Handle empty time coordinates
        if time_values.size == 0:
            continue

        # Find the first unsorted pair
        is_sorted_within = np.all(time_values[:-1] <= time_values[1:])

        if not is_sorted_within:
            print(f"Error: 'time' coordinate is not sorted within dataset at index {i}.")

            # Find the first non-monotonic pair
            problem_indices = np.where(time_values[:-1] > time_values[1:])[0]
            if problem_indices.size > 0:
                first_problem_index = problem_indices[0]
                problem_pair = (time_values[first_problem_index], time_values[first_problem_index + 1])
                print(
                    f"  First non-monotonic pair found at indices {first_problem_index} and {first_problem_index + 1}:")
                print(f"  Value at index {first_problem_index}: {problem_pair[0]}")
                print(f"  Value at index {first_problem_index + 1}: {problem_pair[1]}")
                print(f"  Full time values for this dataset: {time_values}")
            return False

    # 3. Check across datasets
    # Create a single array of all time coordinates
    all_times = np.concatenate([ds.coords['time'].values for ds in datasets])

    # Check if this combined array is monotonically increasing
    is_sorted_across = np.all(all_times[:-1] <= all_times[1:])
    if not is_sorted_across:
        print("Error: 'time' coordinate is not sorted across all datasets.")

        # Find the first non-monotonic pair
        problem_indices = np.where(all_times[:-1] > all_times[1:])[0]
        if problem_indices.size > 0:
            first_problem_index = problem_indices[0]
            problem_pair = (all_times[first_problem_index], all_times[first_problem_index + 1])
            print(
                f"  First non-monotonic pair found at indices {first_problem_index} and {first_problem_index + 1} of the combined array:")
            print(f"  Value at index {first_problem_index}: {problem_pair[0]}")
            print(f"  Value at index {first_problem_index + 1}: {problem_pair[1]}")

        return False

    print("Success: 'time' coordinate is monotonically increasing across all datasets.")
    return True
