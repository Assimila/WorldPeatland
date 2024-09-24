from osgeo import gdal, osr
from osgeo import gdal_array
import numpy as np
from datetime import datetime as dt
import xarray as xr

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
        
    for i in range(1,opn.RasterCount + 1):
            
        rst = opn.GetRasterBand(i)
        meta = rst.GetMetadata()
        
        # following fill in with the corresponding format 
        # 'time' check the metadata of the tiff to see what they call
        # could also be 'RANGEBEGINNINGDATE'
        # the time data
        x = meta[time]
        # also check the metadata to see how is the format of datetime data
        dt_format = '%Y-%m-%dT%H:%M:%S.000000000'
        #dt_format = '%Y-%m-%d %H:%M:%S'
        #dt_format = '%Y-%m-%d'
        t = dt.strptime(x, dt_format)
            
        # append it to the list
        dts.append(t)
                        
    # save the last osegeodataset for its srs
    saved_opn = opn
        
    # open the array
    arr = opn.ReadAsArray()
    
    return arr, dts, saved_opn



def gdal_stack_dt(lt, time):
    
    '''
    gdal_stack_dt function will open geotiffs files and concatenate the dataset, 
    set the time as datetime 
    
    INPUTS:
        - lt (list) - list containing all the geotiffs files to be concatenated
        - time (string) - check how the time variable is written in the tiff metadata
    
    OUTPUTS:
        - stacked_arr (np.array) - stacked array containing all the layers of the 
            input
        - dts (list) - list of the datetimes 
        - saved_opn (osegeo gdal dataset) - saved dataset for its srs 
    '''
    
    # Create empty lists
    ARRAYS_RESHAPED = []
    dts = []
    
    # loop throught the files
    for e in lt:
        # Check if the input is a str which would be the tif file 
        # otherwise it is already an opened gdal dataset
        
        #print(e)
        
        if type(e) == str:
            # open the Dataset
            opn = gdal.Open(e)
            
        else:
            opn = e
            
        # Gdal counts from 1 
        for i in range(1,opn.RasterCount + 1):
            
            rst = opn.GetRasterBand(i)
            meta = rst.GetMetadata()
            # following fill in with the corresponding format 
            # 'time' check the metadata of the tiff to see what they call
            # the time data            
            x = meta[time]
            # also check the metadata to see how is the format of datetime data
            # dt_format = '%Y-%m-%dT%X.000000000'
            dt_format = '%Y-%m-%d %H:%M:%S'  # '2017-03-05 10:10:21'
            t = dt.strptime(x, dt_format)
            
            # append it to the list
            dts.append(t)
                        
        # save the last osegeodataset for its srs
        saved_opn = opn
        
        # open the array
        arr = opn.ReadAsArray()
        
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

    return stacked_arr, dts, saved_opn


def create_xarr(opn, var_name, arr, dts):
    '''
    create_xarr function will create an xarray 
    INPUTS:
        - opn (osgeo gdal dataset) - 
        - var_name (string) - name of the variable, or the band or the information
            stored in the pixels of the tif, or in the array 
        - arr (np.array) - contains all the data values to be stored from a tiff 
            to an xarray
    OUTPUTS:
        - ds (xarray) - dataset of the xarray
    '''
    # create the x and y list of coordinates 
    # GetGeotransform gets me the corner coordinates of the tiff 
    # i.e. (564550.0, 10.0, 0.0, 5931390.0, 0.0, -10.0)
    params = opn.GetGeoTransform()  # params is a tuple 
    xs = np.array([params[0]+(params[1]*i) + (params[1]/2)  for i in np.arange(opn.RasterXSize)])
    # params[0] is the top left point x/lon coordinate value
    # prams[1] is the length along the x axis of 1 pixel
    # RasterXSize is the total number of pixels along the x-axis (later it would be the size of the whole tiff of xarray)
    ys = np.array([params[3]+(params[5]*i) + (params[5]/2) for i in np.arange(opn.RasterYSize)]) 
    # params[3] is the same point top left but now its y/lat coordinate value 
    # params[5] is the length or step to reach the second point along the y axis of one pixel 
    # it is - because you are going downward the y-axis or latitude line 

    variable_name = var_name

    ds = xr.Dataset(data_vars = {variable_name:(('time','latitude', 'longitude'),arr)},
                   coords={'time': dts,
                          'latitude': ys,
                          'longitude': xs})
    return ds

def create_coord_list(opn):
    '''Path the osgeo gdal database that you need to get the references xs and ys'''
    params = opn.GetGeoTransform()
    xs = [params[0]+(params[1]*i) for i in np.arange(opn.RasterXSize)]
    ys = [params[3]+(params[5]*i) for i in np.arange(opn.RasterYSize)]
    
    x = [params[0]+(params[1]*i) + (params[1]/2) for i in np.arange(opn.RasterXSize)]
    y = [params[3]+(params[5]*i) + (params[5]/2) for i in np.arange(opn.RasterYSize)]
    
    print(f'old: {xs} , {ys}')
    print('new: ', x,y)
    
    return xs,ys


def reproject_image(source_img, target_img, clip_shapefile = None, no_data_val = -9999):
 
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
    if type(source_img) == str:
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
                    cropToCutline=True,dstNodata = no_data_val)
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
                    cropToCutline=True,dstNodata = no_data_val)
 
        else:
            g = gdal.Warp('', source_img, format='MEM',
                      outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                      dstSRS=dstSRS)
    return g       