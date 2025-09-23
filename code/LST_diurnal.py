import xarray as xr
import numpy as np

import sys
from WorldPeatland.code.gdal_sheep import create_xarr, gdal_dt, _get_FillValue, get_NoDataValue
from WorldPeatland.code.save_xarray_to_gtiff_old import save_xarray_old
from WorldPeatland.code.utils import proj4_extract

def main(site):

    path = f'/wp_data/sites/{site}/MODIS/timeSeries/'
    # LST DAY linear.smoothn.descaled
    fpath = path + 'MYD11A1.061._LST_Day_1km.linear.smoothn.0.5.descaled.tif'
    arr, dts, opn = gdal_dt(fpath)
    xr_lstDay = create_xarr(opn, 'LST', arr, dts)

    # LST NIGHT linear.smoothn.descaled
    fpath = path + 'MYD11A1.061._LST_Night_1km.linear.smoothn.0.5.descaled.tif'
    arr, dts, opn = gdal_dt(fpath)
    xr_lstNight = create_xarr(opn, 'LST', arr, dts)

    opn_ref = opn

    # Actual smoothened Observations of LSTs
    # Difference between day and night LST
    xr_lstDay['LST_diurnal'] = xr_lstDay['LST'] - xr_lstNight['LST']

    proj4_str = proj4_extract(opn)
    xr_lstDay.attrs['crs'] = proj4_str

    NoDataValue = get_NoDataValue(opn)

    # Add FillValue to attributes
    xr_lstDay.attrs['_FillValue'] = NoDataValue
    output_fname = path + 'MYD11A1.061._LST_Diurnal_1km.linear.smoothn.0.5.descaled.tif'
    # save the xarray into a tif file
    save_xarray_old(output_fname, xr_lstDay, 'LST_diurnal', opn_ref.GetGeoTransform())


    trend_ds = xr_lstDay['LST_diurnal'].rolling(time=int(365), min_periods=1, center=True).mean()
    ds_diur_dtr = trend_ds.to_dataset(name='LST_diurnal')

    ds_diur_dtr.attrs['crs'] = proj4_str

    # Add FillValue to attributes
    ds_diur_dtr.attrs['_FillValue'] = NoDataValue
    output_fname = path + 'MYD11A1.061._LST_Diurnal_1km.linear.smoothn.0.5.descaled.detrended.tif'

    # save the xarray into a tif file
    save_xarray_old(output_fname, ds_diur_dtr, 'LST_diurnal', opn_ref.GetGeoTransform())


    # LST DAY linear.smoothn std dev
    fpath = path + 'MYD11A1.061._LST_Day_1km.linear.smoothn.0.5_qa_weighted_std_dev.tif'
    arr, dts, opn = gdal_dt(fpath)
    xr_lstDay = create_xarr(opn, 'LST', arr, dts)

    # LST NIGHT linear.smoothn std dev
    fpath = path + 'MYD11A1.061._LST_Night_1km.linear.smoothn.0.5_qa_weighted_std_dev.tif'
    arr, dts, opn = gdal_dt(fpath)
    xr_lstNight = create_xarr(opn, 'LST', arr, dts)

    uncertainty_diur = xr.Dataset()
    uncertainty_diur['LST'] = np.sqrt(xr_lstDay['LST']**2 + xr_lstNight['LST']**2)

    # set 'crs'
    uncertainty_diur.attrs['crs'] = proj4_str

    # get fill value
    _FillValue = _get_FillValue(opn)

    # Add FillValue to attributes
    uncertainty_diur.attrs['_FillValue'] = _FillValue

    # Rename dimensions
    uncertainty_diur = uncertainty_diur.rename({'time': 'RANGEBEGINNINGDATE'})


    output_fname = path + 'MYD11A1.061._LST_Diurnal_1km.linear.smoothn.0.5_std_dev.tif'

    # save the xarray into a tif file
    save_xarray_old(output_fname, uncertainty_diur, 'LST', opn_ref.GetGeoTransform())

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python script.py <site_name>")  # the user has to input one argument
    else:
        site_name = sys.argv[1]
        main(site_name)


