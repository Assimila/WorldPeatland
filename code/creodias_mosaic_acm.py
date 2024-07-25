
import os
import requests
from glob import glob
from requests.utils import requote_uri
from osgeo import gdal, ogr, osr
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np 
from lightgbm import Booster
import json
from pyproj import Transformer
from calendar import monthrange
import sys
sys.path.insert(0, '/workspace/WorldPeatland/code/')
from gdal_sheep import *
from save_xarray_to_gtiff_old import *
import matplotlib.pyplot as plt

def transform_coordinate(x: float, y: float,
    output_crs: str,
    input_crs="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"):
    """
    Transforms a x,y coordinate in an input coordinate reference
    system (CRS) to a x, y coordinate in an user-defined CRS
    Both CRS' must be a proj4 string. Default input is
    Lat/Lon EPSG:4326: https://spatialreference.org/ref/epsg/wgs-84/
    :param x: x element of the coordinate
    :param y: y element of the coordinate
    :param output_crs: output crs
    :param input_crs: input crs
    :return (proj_x, proj_y): tupple with projected coordinates to
                              output CRS
    """
    # Create transformer
    transformer = Transformer.from_crs(input_crs, output_crs)
    # Transform coordinates
    proj_x, proj_y = transformer.transform(x, y)

    return (proj_x, proj_y)

def get_extent(fname):
    """
    Get extent from GeoJSON file as:
    xmin, xmax, ymin, ymax
    """
    d = ogr.Open(fname)
    l = d.GetLayer()
    return l.GetExtent()

def get_polygon(fname):
    """
    Get POLYGON string from GeoJSON coordinates
    """
    with open(fname) as f:
        geojson = json.load(f)

    polygon = ""
    for coordinate in geojson['features'][0]['geometry']['coordinates'][0]:
        _coordinate = " ".join(str(x) for x in coordinate)
        polygon = f"{polygon} {_coordinate}, "

    polygon = polygon[0:-2]

    return polygon

def create_dir(path):
    """
    Create directory
    """
    os.makedirs(path, exist_ok=True)

    return None


def get_spatial_reference(img_path):
    """
    get spatial reference from input img path
    """
    
    opn = gdal.Open(img_path)

    if opn is None:
        raise FileNotFoundError(f"Unable to open {image_path}")
    proj_wkt = opn.GetProjection()
    srs = osr.SpatialReference(wkt=proj_wkt)

    return srs

def reproject_srs(img_path, output_path, target_srs):
    opn = gdal.Open(img_path)

    if opn is None:
        raise FileNotFoundError(f"Unable to open {image_path}")
    
    # Reproject the image 
    gdal.Warp(output_path, opn, dstSRS=target_srs.ExportToWkt())
    

def check_srs(img_list, output_dir):
    """
    check srs of all images in the list, if different reproject
    if not return True

    INPUTS
        - img_list - list of str paths to the images
        - output_dir - str to the path of the site specific 
            S2 repository
    """
    # set the target image srs to be the first image of the month  
    reference_srs = get_spatial_reference(img_list[0])
    reproj_inputs_dirs = []
    for img_path in img_list:
        
        # get img_path srs
        srs = get_spatial_reference(img_path)
        if not reference_srs.IsSame(srs):
            reproj_img_path = Path(img_path).name
            reproj_img_path = reproj_img_path.replace(".jp2", "_reprojected.jp2")
            reproj_img_path = os.path.join(output_dir, reproj_img_path)
            
            # run the reprojection function 
            reproject_srs(img_path, reproj_img_path, reference_srs)
            reproj_inputs_dirs.append(reproj_img_path)
            
        else:
            reproj_inputs_dirs.append(img_path)
    
    return reproj_inputs_dirs
    

def create_mosaic_subset(input_dirs, output_dir, extent, band):
    
    for i in range(len(input_dirs)):
        fname = glob(input_dirs[i])
        if len(fname) > 0:
            input_dirs[i] = fname[0]
        #TODO Check that files exist

    create_dir(output_dir)
    fname = os.path.splitext(os.path.basename(input_dirs[0]))[0]

    output_fname = os.path.join(output_dir, fname) 
    
    #TODO reproject to the crs with the highest percentage of coverage

    # Get extent in native CRS
    dst_crs = get_crs(input_dirs[0])
    minX, minY = transform_coordinate(extent[0], extent[2], 
            output_crs=dst_crs)
    maxX, maxY = transform_coordinate(extent[1], extent[3],
            output_crs=dst_crs)

    extent_native_crs = (minX, minY, maxX, maxY)
    output_fname = f'{output_fname}.vrt'

    # check that the file is the same UTM zone 
    input_dirs = check_srs(input_dirs, output_dir)


    if len(input_dirs) == 1:

        options = gdal.WarpOptions(format='VRT',
                outputBounds=extent_native_crs)
        vrt = gdal.Warp(output_fname, input_dirs, options=options)

        vrt = None
        del(vrt)

    else:
        # check all imgs have the same projection
        # pick for now img[0] as reference
        # run check/reprojection
        inputs_dirs = check_srs(input_dirs, output_dir)

        # Create mosaic and subset
        # Create a mosaic directory inside every band to avoid picking 
        # the mosaic files at a later stage instead of the actual clipped
        # vrt (will get an error of dataset with different shapes)
        # cannot delete the mosaics because they are the pointers to the
        # clipped vrts created later
        mosaic_dir = output_dir + '/mosaics'
        create_dir(mosaic_dir)
        
        mosaic_fname = os.path.join(mosaic_dir, fname)
        mosaic_fname = f'{mosaic_fname}_mosaic.vrt'
        mosaic = gdal.BuildVRT(mosaic_fname, inputs_dirs)
        mosaic.FlushCache()
        mosaic = None

        options = gdal.WarpOptions(format='VRT', 
                outputBounds=extent_native_crs)
        
        vrt = gdal.Warp(output_fname, mosaic_fname, options=options)
        vrt = None
        del(vrt)

    return output_fname


def reproject_to_B02(fname, target_img):
    """
    reproject all images to B02 resolution to be able to stack the arrays 

    INPUTS

        - target_img - str B02 file path
    """
    
    g = reproject_image(fname, target_img, clip_shapefile = None, no_data_val = -9999)
    
    return g


def get_crs(fname):
    """
    Get CRS in WKT string from a single file
    """
    d = gdal.Open(fname)
    proj = d.GetProjection()
    return proj

def get_geotransform(fname):
    """
    Get geotransform from a single file 
    """
    d = gdal.Open(fname)
    geotransform = d.GetGeoTransform()
    return geotransform 

def create_daily_vrts(S3Paths, year, month, days, extent, product='S2_TOA'):
    """
    Create mosaics for daily set of Sentinel-2 acquisitions
    """
    # Dictionary to store all ouputs per dataset-band
    outputs = {}

    # Find all images for a particular day
    for day in range(1, days+1):
        date=f"{year}{month:02}{day:02}"
        images = []
        for img in S3Paths:
            if img.find(date) > 0:
                images.append(img)

        if len(images) == 0:
            images = []
            continue
        
        # If there is a single image just subset 
        for band in datasets:

            img_path = f'GRANULE/*/IMG_DATA/*{band}*.jp2'
            
            output_dir = os.path.join(OUTPUTDIR, 'datacube',
                     product, band, 'VRTs')

            images_path = []
            for i in range(len(images)):
                images_path.append(os.path.join(images[i], img_path))
            output_fnames = create_mosaic_subset(images_path,
                    output_dir, extent, band)

            if band in outputs:
                outputs[band].append(output_fnames)
            else:
                outputs[band] = [output_fnames]

    return outputs

def get_timesteps(outputs):
    """
    get the exact timestep date from the vrt file names
    """
    timesteps = []
    for i, x in enumerate(outputs['B02']):
        time = os.path.basename(x).split('_')[1]
        timesteps.append(time)

    return timesteps


def get_flist(t, product='S2_TOA'):
    """
    get a list of the file paths of all bands for one time steps
    INPUT:
        - t - str of one time step as it appears in the vrt filename
    OUTPUT:
        - fl - list of vrt file names
    """

    fl = []

    for band in datasets:
        # create the list of paths for all the bands for one timestep
        band_dir = os.path.join(OUTPUTDIR, 'datacube', product, band)
        name = glob(band_dir + f'/VRTs/*{t}*.vrt')[0]
        fl.append(name)

    return fl


def run_acm(outputs, product='S2_TOA'):
     
    """
    Loop over all timesteps, create acm layer for each timestep, 
    create xarray per time step and then save the xarray into a geotiff

    INPUT: 
        - outputs (dict) - keys are all the bands necessary to perform cprob, 
            values are lists of the paths to the vrt for every timestep
    OUTPUT:

        - outputs (dict) - key: ACM 
                            Values: paths to the ACM tiff file paths per timestep
    """
    outputs_acm = {}

    output_dir = os.path.join(OUTPUTDIR, 'datacube', product, 'ACM')
    create_dir(output_dir)
    timesteps = get_timesteps(outputs)
    
    for i, t in enumerate(timesteps):
        # loop over all time steps for the month 
        fl = get_flist(t, product='S2_TOA')
        # calculate the acm array 
        cprob_arr = calculate_acm(fl)

        # get date and coordinates from B02 vrt file
        opn = gdal.Open(fl[1])
        # get spatial reference from opn dataset
        proj_wkt = opn.GetProjection()
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(proj_wkt)
        proj4_string = spatial_ref.ExportToProj4()

        dt = str(pd.to_datetime(t, format='%Y%m%dT%H%M%S'))
        # create a datetime index 
        dt = pd.date_range(dt, periods=1)
        # create the xarray 
        ds = create_xarr(opn, 'cprob', cprob_arr[np.newaxis, :, :], dt)
        ds.attrs['crs'] = proj4_string 
        # save the xarray as geotiff 
        fname = output_dir + f'/ACM_{t}.tif'
        save_xarray_old(fname, ds, 'cprob')

        if 'ACM' in outputs_acm:
            outputs_acm['ACM'].append(fname)
        else:
            outputs_acm['ACM'] = [fname]

    return outputs_acm


def calculate_acm(flist):
    
    """
    Calculate acm for one time step
    
    INPUT:
        - flist - list of the file paths for each required band
            for one time step
    OUTPUT:
        - reshaped_cprob - (numpy array) with the shape od B02 clipped vrt
            containing the calculated cloud probability mask
    """
    ls= []
    for path in flist:

        # regrid flist[1] is the B02 path 
        g = reproject_to_B02(path, flist[1])
        # seems like the arr is flipping the x and y size position??
        arr = g.ReadAsArray()
        ls.append(arr)
    
    # stack the arrays in the list
    # stack_arr shape (number of bands, Xsize, Ysize)
    # in this case for this cloud porbability model number of
    # bands required is 10, Xsize and Ysize should be the same as a B02
    # image because all bands are regridded to follow B02
    stack_arr = np.stack(ls, axis = 0)
    
    # remove offset and multiply by scaling factor (/10000)
    _stack_arr = (stack_arr - 1000) / 10000
    shp = _stack_arr.shape

    # per pixel array 
    # pixels array shape (Xsize * Ysize, nb of bands)
    pixels = _stack_arr.reshape([shp[0], shp[1]*shp[2]]).T

    # apply the cloudless probability on the 2d array  
    pmodel = Booster(model_file=(
        "/workspace/WorldPeatland/code/pixel_s2_cloud_detector_lightGBM_v0.1.txt"))
    # cprob array shape is (Xsize * Ysize) 1d array
    cprob = pmodel.predict(pixels)
        
    # reshape the cloud probability 1d array into 2d array
    # cprob_2d array shape is (Xsize, Ysize) of an initial B02 image
    cprob_2d = cprob.reshape([shp[1],shp[2]])
    
    return cprob_2d



def create_monthly_cogs(ouputs_acm, year, month, product = 'S2_TOA'):
    """
    Create monthly DataCube COGs
    """
    
    for dataset in outputs_acm:
         # Get unique filename and check it doesnt exist in tmp_path
        f = tempfile.NamedTemporaryFile(mode='w+b', delete=True,
                                        dir='/tmp', suffix=".vrt")
        output_vrt_fname = f.name

        build_options = gdal.BuildVRTOptions(separate=True)
        vrt = gdal.BuildVRT(output_vrt_fname, outputs_acm[dataset],
                options=build_options)

        for i in range(vrt.RasterCount):
            _time = outputs_acm[dataset][i]
            _time = os.path.basename(_time)
            if dataset == 'MSK_CLDPRB_20m':
                _time = _time.split('_')[3]
            else:
                _time = _time.split('_')[1]
            
            # Remove the tif extension 
            if '.' in _time:
                _time = _time.split('.')[0]

            _time = str(pd.to_datetime(_time, format='%Y%m%dT%H%M%S'))

            band = vrt.GetRasterBand(i+1)
            band.SetMetadataItem('add_offset', '0')
            band.SetMetadataItem('data_var', product)
            band.SetMetadataItem('fill_value', '999')
            band.SetMetadataItem('product', product)
            band.SetMetadataItem('scale_factor', '1.0')
            band.SetMetadataItem('time', _time)
            band.SetMetadataItem('version', 'Sentinel-2_ACM')

        vrt = None ; del(vrt)

        translate_options = gdal.TranslateOptions(format='GTiff')

        # Get output dir from first VRT
        output_dir = Path(outputs_acm[dataset][0])
        output_dir = str(output_dir.parent.absolute())
        # Output file name
        output_cog_fname = f'{product}_{dataset}_{year}-{month:02}.tif'
        output_cog_fname = os.path.join(output_dir, output_cog_fname)

        tmp_ds = gdal.Translate(output_cog_fname, output_vrt_fname,
                options=translate_options)

        # Cleant tmp variables
        tmp_ds = None ; del(tmp_ds)
        f = None ; del(f)
    
        # remove all single timestep ACM tif files
        for f in outputs_acm[dataset]:
            os.remove(f)



# - Creates subset/mosaic for every day when acquisitions have been found
# - Creates monthly DQ COGs for every month
# =======================================================================#

datasets =  ['B01','B02', 'B04', 'B05', 'B08',
            'B8A', 'B09', 'B10', 'B11', 'B12']
            
cloud_cover_le = 30


OUTPUTDIR = '/wp_data/sites/Norfolk/Sentinel/MSIL1C'
create_dir(OUTPUTDIR)

geojson_fname = '/workspace/WorldPeatland/sites/Norfolk.geojson'
extent = get_extent(geojson_fname) 
polygon = get_polygon(geojson_fname)


# this link is taken from creodias website  https://explore.creodias.eu/search 
# Sentinel_2: platform A & B, S2MSI1C, and collection 1 
# collection 1 gives only the L1_N500 products 

url_start = (f"https://datahub.creodias.eu/odata/v1/Products?$filter="
             f"((Attributes/OData.CSC.DoubleAttribute/any(i0:i0/Name eq %27cloudCover%27 and i0/Value le {cloud_cover_le})) and ")

url_end = (f"(Online eq true) and "
           f"(OData.CSC.Intersects(Footprint=geography%27SRID=4326;POLYGON%20(("
           f"{polygon}"
           f"))%27)) and "
           f"(((((Collection/Name%20eq%20%27SENTINEL-2%27)%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27platformSerialIdentifier%27%20and%20i0/Value%20eq%20%27A%27))%20or%20(Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27platformSerialIdentifier%27%20and%20i0/Value%20eq%20%27B%27))))%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27productType%27%20and%20i0/Value%20eq%20%27S2MSI1C%27))))%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27processorVersion%27%20and%20i0/Value%20eq%20%2705.00%27))%20or%20(Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27processorVersion%27%20and%20i0/Value%20eq%20%2705.09%27)))))))))&$expand=Attributes&$expand=Assets&$orderby=ContentDate/Start%20asc&$top=20")


for year in range(2017, 2024+1):
    for month in range(1, 12+1):
        start_date = f'{year}-{month:02}-01T00:00:00.000Z'
        end_day = monthrange(year, month)[1]
        end_date = f'{year}-{month:02}-{end_day:02}T23:59:59.999Z'

        url = (f"{url_start}"
               f"(ContentDate/Start ge {start_date} and ContentDate/Start le {end_date}) and "
               f"{url_end}")

        # Encode URL
        url_encoded = requote_uri(url)

        # Remove unnecessasary characters from encoded URL
        url_encoded_cleared = url_encoded.replace('%0A', '')
        # Obtain and print the response
        response = requests.get(url_encoded_cleared)
        response = response.json()
        S3Paths = []
        for i, element in enumerate(response['value']):
            S3Paths.append(element['S3Path'])

            image_name = os.path.basename(element['S3Path'])
            new_dir = os.path.join(OUTPUTDIR, image_name)
            try:
                os.symlink(element['S3Path'], new_dir,
                    target_is_directory=True)

            except FileExistsError:
                print(f"{element['S3Path']} already exists")
                print()
        # Create daily VRTs
        outputs = create_daily_vrts(S3Paths, year, month, end_day, extent)
        # Calculate ACM
        if not outputs :
            continue
        else:
            outputs_acm = run_acm(outputs, product='S2_TOA')
            # Check srs of the time steps in a month 
            output_dir = os.path.join(OUTPUTDIR, 'datacube', 'S2_TOA', 'ACM')
            outputs_acm['ACM'] = check_srs(outputs_acm['ACM'], output_dir)

            # Create monthly COGs
            create_monthly_cogs(outputs_acm, year, month)
