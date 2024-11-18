import os.path
from glob import glob
import re
from WorldPeatland.code.downloader_wp_test import *
from WorldPeatland.code.gdal_sheep import *
from WorldPeatland.code.save_xarray_to_gtiff_old import *

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def gdal_dt(e, time):
    '''
    gdal_dt function will open the tif file as an osegeo gdal dataset

    INPUTS:
        - e (str or tiff) - path the tiff file or the gdal dataset you want to
            open and save its datetimes
        - time (string) - check how the time variable is written in the tiff metadata
    Outputs:
        - arr (np.array) - return arr of the gdal dataset
        - dts (list) - list of the datetimes
        - saved_opn (osegeo gdal dataset) - saved dataset for its srs
    '''

    # Create an empty list to store the datatimes
    dts = []

    # Check if the input is a str which would be the tif file
    # otherwise it is already an opened gdal dataset
    if type(e) == str:
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
        x = meta[time]
        # also check the metadata to see how is the format of datetime data
        dt_format = '%Y-%m-%dT%H:%M:%S.000000000'
        # dt_format = '%Y-%m-%d %H:%M:%S'
        # dt_format = '%Y-%m-%d'
        t = dt.strptime(x, dt_format)

        # append it to the list
        dts.append(t)

    # save the last osegeodataset for its srs
    saved_opn = opn

    # open the array
    arr = opn.ReadAsArray()

    return arr, dts, saved_opn


def extract_data_var(opn):
    """
    from an osgeo gdal object extract from in raster band metadata the 'data_var' name

    INPUT
        - opn (osegeo gdal_obj) -
    OUTPUT
        - data_var (str) - dara variable name in the current tiff EVI, LAI, Albedo....
    """

    # Get into the first raster band of this tiff
    rb = opn.GetRasterBand(1)  # could be any band number #rb is a gdal raster band obj
    mtd = rb.GetMetadata()  # mtd is a dict
    return mtd['data_var']


def create_xarr_from_tif_path(tif_path):
    """
    Create the xarray from the tif path

    INPUT
        - tif_path (str) - path to the tif file of the data variable

    OUTPUT
        - ds (xarray.Dataset) - xarray dataset of the tif data variable
    """

    arr, dts, saved_opn = gdal_dt(tif_path, 'time')
    # Extract name of the variable from the gdal obj
    var_name = extract_data_var(saved_opn)
    ds = create_xarr(saved_opn, var_name, arr, dts)
    return ds, var_name


def extract_crs_from_tif(tif_path):
    """
    Extract the crs from the tif_path using command line

    INPUT
        - tif_path (str) - string path to the tif file

    OUTPUT
        - proj4_string (str) - of the crs coordinate reference system
    """
    # get crs or the proj4 str from the monthly-tif using the command line of gdalinfo
    command = ['gdalinfo', tif_path, '-proj4']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # use regex to find the string in the results
    proj4_match = re.search(r"PROJ\.4 string is:\n'(.*?)'", result.stdout)
    proj4_string = proj4_match.group(1)
    return proj4_string


def main(ts_path, output_dir):

    # list of tif_paths for the non detrended data variable tif products only
    for path in (glob.glob(f'{ts_path}/*descaled.tif')):

        ds, var_name = create_xarr_from_tif_path(path)

        # remove the wing years which only contains 6 months of the data
        years = ds.coords['time'].dt.year
        first_year = years.min().values
        last_year = years.max().values
        ds = ds.sel(time=(years > first_year) & (years < last_year))

        # Calculate climatology
        ds_mean = ds.groupby("time.dayofyear").mean()
        ds_std = ds.groupby("time.dayofyear").std()

        # get the years from the data
        grouped_by_year = ds.time.groupby("time.year")  # type xarray.core.groupby.DataArrayGroupBy
        anomalies_list = []
        for i, (year, _time) in enumerate(grouped_by_year):
            # Get time series for year (np.int64)
            ds_year = ds.sel(time=_time)  # ds_year type is xarr dataset

            # assign dayofyear coordinates to ds_year
            ds_year = ds_year.assign_coords(dayofyear=ds_year.time.dt.dayofyear)

            # select the corresponding doy for mean and std
            ds_mean_sel = ds_mean.sel(dayofyear=ds_year.dayofyear)
            ds_std_sel = ds_std.sel(dayofyear=ds_year.dayofyear)

            anomalies_year = (ds_year - ds_mean_sel[var_name].data) / ds_std_sel[var_name].data
            anomalies_list.append(anomalies_year)

        ds_anomalies_all_years = xr.concat(anomalies_list, dim='time')

        proj4_string = extract_crs_from_tif(path)
        # set proj4 str as an attribute to the xarray so that it can be saved as a tiff
        ds_mean.attrs['crs'] = proj4_string
        ds_std.attrs['crs'] = proj4_string
        ds_anomalies_all_years.attrs['crs'] = proj4_string

        # output_path
        output_path_mean = os.path.join(output_dir, f'{var_name}_climatology_mean.tif')
        output_path_std = os.path.join(output_dir, f'{var_name}_climatology_std.tif')
        output_path_anomalies = os.path.join(output_dir, f'{var_name}_anomalies.tif')

        # edited version of save_xarray_to_gtiff
        save_tiff(output_path_mean, ds_mean, var_name)
        save_tiff(output_path_std, ds_std, var_name)
        save_tiff(output_path_anomalies, ds_anomalies_all_years, var_name)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <ts_path>")  # the user has to input two arguments
    else:
        # provide the path to the time_series created from MODIS data
        ts_path = sys.argv[1]
        output_dir = sys.argv[2]
        main(ts_path, output_dir)