from osgeo import gdal, osr
from osgeo import gdal_array
import dask.array as da
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_dst_dataset(dst_img, cols, rows, layers, dtype, proj, gt):
    """
    Create a GDAL data set in Cloud Optimized GeoTIFF (COG) format
    :param dst_img: Output filenane full path
    :param cols: Number of columns
    :param rows: Number of rows 
    :param layers: Number of layers
    :param dtype: GDAL type code
    :param proj: Projection information in WKT format
    :param gt: GeoTransform tupple
    :return dst_ds: GDAL destination dataset object
    """
    gdal.UseExceptions()
    try:
        # Default driver options to create a COG
        driver = gdal.GetDriverByName('GTiff')
        driver_options = ['COMPRESS=DEFLATE',
                          'BIGTIFF=YES',
                          'PREDICTOR=1',
                          'TILED=YES',
                          'COPY_SRC_OVERVIEWS=YES']

        # Create driver
        dst_ds = driver.Create(dst_img, cols, rows, layers,
                               dtype, driver_options)

        # Set cartographic projection
        dst_ds.SetProjection(proj)
        dst_ds.SetGeoTransform(gt)

    except Exception as err:
        err_level = gdal.GetLastErrorType()
        err_msg = gdal.GetLastErrorMsg()

        if err_level >= gdal.CE_Warning:
            # Stop using GDAL exceptions
            gdal.DontUseExceptions()
            raise RuntimeError(f"GDAL Error {err_level}: {err_msg}")

    gdal.DontUseExceptions()
    return dst_ds

def get_resolution_from_xarray(xarray):
    """
    Method to create a tuple (x resolution, y resolution) in x and y
    dimensions.

    :param xarray: xarray with latitude and longitude variables.

    :return: tuple with x and y resolutions
    """

    x_res = xarray.longitude.values[1] - xarray.longitude.values[0]
    y_res = xarray.latitude.values[1] - xarray.latitude.values[0]

    return (x_res, y_res)
  
def save_xarray_old(fname, xarray, data_var, gt=None):
    """
    Saves an xarray dataset to a Cloud Optimized GeoTIFF (COG)
    :param fname: Full path of file where to save the data
    :param xarray: xarray Dataset
    :param data_var: Data variable in the xarray dataset
    :param gt: Optional GeoTransform array
    """

    # Create GeoTransform - perhaps the user requested a
    # spatial subset, therefore is compulsory to update it

    # GeoTransform -- case of a "north up" image without
    #                 any rotation or shearing
    #  GeoTransform[0] top left x
    #  GeoTransform[1] w-e pixel resolution
    #  GeoTransform[2] 0
    #  GeoTransform[3] top left y
    #  GeoTransform[4] 0
    #  GeoTransform[5] n-s pixel resolution (negative value)

    # Create tmp xarray DataArray
    
#     _xarray = getattr(xarray, data_var)

    _xarray = xarray[data_var]

    if gt == None:

        x_res, y_res = get_resolution_from_xarray(_xarray)

        gt = (_xarray.longitude.data[0] - (x_res / 2.),
              x_res, 0.0,
              _xarray.latitude.data[0] - (y_res / 2.),
              0.0, y_res)


    # Coordinate Reference System (CRS) in a PROJ4 string to a
    # Spatial Reference System Well known Text (WKT)
    crs = xarray.attrs['crs']
    srs = osr.SpatialReference()
    srs.ImportFromProj4(crs)
    proj = srs.ExportToWkt()

    # Get GDAL datatype from NumPy datatype
    if _xarray.dtype == 'bool':
        dtype = gdal.GDT_Byte
    else:
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(_xarray.dtype)

    # Dimensions
    layers, rows, cols = _xarray.shape

    # Create destination dataset
    dst_ds = get_dst_dataset(dst_img=fname, cols=cols, rows=rows,
                 layers=layers, dtype=dtype, proj=proj, gt=gt)

    # keys = ['_FillValue', 'fill_value']
    # for key in keys:
    #     if key in xarray.attrs:
    #         _FillValue = float(xarray.attrs[key])
    #         break
    # else:

        # _FillValue = float(np.nan)  # Default to NaN if not specified
        # print("No valid fill value key found in metadata, fill value set as nan.")

    for layer in range(layers):
        dst_band = dst_ds.GetRasterBand(layer + 1)

        # Date
        if 'time' in _xarray.dims:
            dst_band.SetMetadataItem('time',
                    _xarray.time.data[layer].astype(str))

        # Data variable name
        dst_band.SetMetadataItem('data_var', data_var)
        #TODO check what happens if _FillValue not specified

        # _FillValue has to be a float
        # dst_band.SetNoDataValue(_FillValue)

        # Check if data is a Dask array
        if isinstance(_xarray[layer].data, da.Array):
            # Compute the dask array to convert it into a Numpy array
            data_to_write = _xarray[layer].data.compute()
        else:
            data_to_write = _xarray[layer].data

        # Write the data to the gdal band
        dst_band.WriteArray(data_to_write)

    LOG.info('save_xarray_to_gtiff_old successful')

    dst_ds = None  
    
    
def save_tiff(fname, xarray, data_var):
    """
    Saves an xarray dataset to a Cloud Optimized GeoTIFF (COG)
    :param fname: Full path of file where to save the data
    :param xarray: xarray Dataset
    :param data_var: Data variable in the xarray dataset
    """

    # Create GeoTransform - perhaps the user requested a
    # spatial subset, therefore is compulsory to update it

    # GeoTransform -- case of a "north up" image without
    #                 any rotation or shearing
    #  GeoTransform[0] top left x
    #  GeoTransform[1] w-e pixel resolution
    #  GeoTransform[2] 0
    #  GeoTransform[3] top left y
    #  GeoTransform[4] 0
    #  GeoTransform[5] n-s pixel resolution (negative value)

    # Create tmp xarray DataArray
    _xarray = xarray[data_var]

    x_res, y_res = get_resolution_from_xarray(_xarray)

    gt = (_xarray.longitude.data[0] - (x_res / 2.),
          x_res, 0.0,
          _xarray.latitude.data[0] - (y_res / 2.),
          0.0, y_res)

    # Coordinate Reference System (CRS) in a PROJ4 string to a
    # Spatial Reference System Well known Text (WKT)
    crs = xarray.attrs['crs']
    srs = osr.SpatialReference()
    srs.ImportFromProj4(crs)
    proj = srs.ExportToWkt()

    # Get GDAL datatype from NumPy datatype
    if _xarray.dtype == 'bool':
        dtype = gdal.GDT_Byte
    else:
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(_xarray.dtype)
    
    # Dimensions
    layers, rows, cols = _xarray.shape

    try:
        # Attempt to create destination dataset
        dst_ds = get_dst_dataset(dst_img=fname, cols=cols, rows=rows,
                                 layers=layers, dtype=dtype, proj=proj, gt=gt)
    except RuntimeError as err:
        print("Runtime error occurred:", err)
        raise
    except Exception as e:
        print("An unexpected error occurred:", e)
        raise


    for layer in range(layers):
        dst_band = dst_ds.GetRasterBand(layer + 1)

        # Date
        if 'time' in _xarray.dims:
            dst_band.SetMetadataItem('time',
                    _xarray.time.data[layer].astype(str))

        # dayofyear
        if 'dayofyear' in _xarray.dims:
            dst_band.SetMetadataItem('dayofyear',
                                     _xarray.dayofyear.data[layer].astype(str))

        # Data variable name
        dst_band.SetMetadataItem('data_var', data_var)

        # Data
        dst_band.WriteArray(_xarray[layer].data)

    dst_ds = None
