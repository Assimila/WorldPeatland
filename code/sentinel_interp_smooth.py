import sys
import os
from osgeo import osr
import logging
from glob import glob
from WorldPeatland.code.utils import create_dir
from WorldPeatland.code.gdal_sheep import gdal_stack_dt, create_xarr
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from WorldPeatland.code.smoothn import smoothn


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

def proj4_extract(opn):

    # get spatial reference from opn dataset
    proj_wkt = opn.GetProjection()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj_wkt)
    proj4_string = spatial_ref.ExportToProj4()

    return proj4_string


def main(site_dir, smoothing_factor=0.5):
    LOG.info(f"Processing site directory: {site_dir}")
    LOG.info(f"Using smoothing factor: {smoothing_factor}")

    mleonn_path = os.path.join(site_dir, 'Sentinel', 'MSIL2A', 'datacube', 'MLEONN', '*')
    paths = glob(mleonn_path)

    if not paths:  # If paths list is empty
        raise FileNotFoundError(f"No MLEONN data products folders found in the specified directory path {mleonn_path}.")

    # Loop over the different MLEONN data products i.e. LAI, cab ...
    for p in paths:
        var_name = os.path.basename(p)
        LOG.info(f'Starting interpolation for {var_name}')

        # get the tif files to interpolate
        tif_files = sorted(glob(os.path.join(p, "*.tif")))

        # Load and stack the files
        stacked_arr, dts, saved_opn = gdal_stack_dt(tif_files)
        data_with_nan = create_xarr(saved_opn, var_name, stacked_arr, dts)

        # Interpolate missing values
        method = 'linear'
        data_interpolated = data_with_nan.interpolate_na(dim='time', method=method)

        # create directory to interpol
        interp_dir = create_dir(p, 'interpolated')
        fname = f'{var_name}.linear.tif'
        output_dir = os.path.join(interp_dir, fname)

        # set extract and set proj 4 to xarray
        proj4_string = proj4_extract(saved_opn)
        data_interpolated.attrs['crs'] = proj4_string
        save_xarray_old(output_dir, data_interpolated, var_name)

        # apply smoothn on the interpolated data
        LOG.info(f'Starting smoothn for {var_name}')
        array_interpolated = data_interpolated[var_name]  # xarray data array
        data_smooth = smoothn(y=array_interpolated, s=smoothing_factor, isrobust=True, axis=0)[0]  # numpy array
        xr_smoothed = create_xarr(saved_opn, var_name, data_smooth, dts)
        fname = f'{var_name}.linear.smoothn.0.5.tif'
        output_dir = os.path.join(interp_dir, fname)
        # set extract and set proj 4 to xarray
        xr_smoothed.attrs['crs'] = proj4_string
        save_xarray_old(output_dir, xr_smoothed, var_name)


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <site_root_data_dir> [smoothing_factor]")
    else:
        site_directory = sys.argv[1]
        smoothing_factor = int(sys.argv[2]) if len(sys.argv) == 3 else 10
        main(site_directory, smoothing_factor)

