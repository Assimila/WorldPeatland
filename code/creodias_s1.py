
import os
import requests
from glob import glob
from requests.utils import requote_uri
from osgeo import gdal, ogr, osr
import tempfile
from pathlib import Path
import pandas as pd
import json
from pyproj import Transformer
from calendar import monthrange

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

def create_mosaic_subset(input_dirs, output_dir, extent, band):
    for i in range(len(input_dirs)):
        fname = glob(input_dirs[i])
        if len(fname) > 0:
            input_dirs[i] = fname[0]
        #TODO Check that files exist

        create_dir(output_dir)
        output_fname = os.path.splitext(os.path.basename(input_dirs[0]))[0]
        output_fname = os.path.join(output_dir, output_fname) 
        output_fname = f'{output_fname}.vrt'

    # Get extent in native CRS
    dst_crs = get_crs(input_dirs[0])
    minX, minY = transform_coordinate(extent[0], extent[2], 
            output_crs=dst_crs)
    maxX, maxY = transform_coordinate(extent[1], extent[3],
            output_crs=dst_crs)

    extent_native_crs = (minX, minY, maxX, maxY)

    if len(input_dirs) == 1:
        # Only create subset
        options = gdal.WarpOptions(format='VRT',
                outputBounds=extent_native_crs)
        vrt = gdal.Warp(output_fname, input_dirs, options=options)
    else:
        # Create mosaic and subset
        options = gdal.BuildVRTOptions(
                outputBounds=extent_native_crs)
        vrt = gdal.BuildVRT(output_fname, input_dirs, options=options)

    vrt = None
    del(vrt)

    return output_fname

def get_crs(fname):
    """
    Get CRS in WKT string from a single file
    """
    d = gdal.Open(fname)
    proj = d.GetGCPProjection()
    return proj

def create_daily_vrts(S3Paths, year, month, days, extent, product='S1_GRD'):
    """
    Create mosaics for daily set of Sentinel-1 acquisitions
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

            img_path = f'measurement/*{band}*.tiff'
            output_dir = os.path.join(OUTPUTDIR, 'datacube',
                     product, band, 'VRTs')

            images_path = []
            for i in range(len(images)):
                images_path.append(os.path.join(images[i], img_path))

            ouput_fnames = create_mosaic_subset(images_path,
                    output_dir, extent, band)

            if band in outputs:
                outputs[band].append(ouput_fnames)
            else:
                outputs[band] = [ouput_fnames]
    return outputs

def create_monthly_cogs(ouputs, year, month, product='S1_GRD'):
    """
    Create monthly DataCube COGs
    """
    for dataset in outputs:
         # Get unique filename and check it doesnt exist in tmp_path
        f = tempfile.NamedTemporaryFile(mode='w+b', delete=True,
                                        dir='/tmp', suffix=".vrt")
        output_vrt_fname = f.name

        build_options = gdal.BuildVRTOptions(separate=True)
        vrt = gdal.BuildVRT(output_vrt_fname, outputs[dataset],
                options=build_options)
        for i in range(vrt.RasterCount):
            _time = outputs[dataset][i]
            _time = os.path.basename(_time)
            _time = _time.split('-')[4]

            _time = str(pd.to_datetime(_time, format='%Y%m%dT%H%M%S'))

            band = vrt.GetRasterBand(i+1)
            band.SetMetadataItem('add_offset', '0')
            band.SetMetadataItem('data_var', product)
            band.SetMetadataItem('fill_value', '999')
            band.SetMetadataItem('product', product)
            band.SetMetadataItem('scale_factor', '1.0')
            band.SetMetadataItem('time', _time)
            band.SetMetadataItem('version', 'Sentinel-2_L2_Sen2Cor')

        vrt = None ; del(vrt)

        translate_options = gdal.TranslateOptions(format='GTiff')
        # Get output dir from first VRT
        output_dir = Path(outputs[dataset][0])
        output_dir = str(output_dir.parent.parent.absolute())
        # Output file name
        output_cog_fname = f'{product}_{dataset}_{year}-{month:02}.tif'
        output_cog_fname = os.path.join(output_dir, output_cog_fname)

        tmp_ds = gdal.Translate(output_cog_fname, output_vrt_fname,
                options=translate_options)

        # Cleant tmp variables
        tmp_ds = None ; del(tmp_ds)
        f = None ; del(f)

# =======================================================================#
# - Queries thi CREODIAS API to search data for specific year and month
# - Creates subset/mosaic for every day when acquisitions have been found
# - Creates monthly DQ COGs for every month
# =======================================================================#

datasets =  ['vh', 'vv']

OUTPUTDIR = '/wp_data/sites/Norfolk/Sentinel/S1GRD'
create_dir(OUTPUTDIR)

geojson_fname = '/workspace/WorldPeatland/sites/Norfolk.geojson'
extent = get_extent(geojson_fname)
polygon = get_polygon(geojson_fname)

url_start = (f"https://datahub.creodias.eu/odata/v1/Products?$filter=")

url_end = (f"(((((Collection/Name%20eq%20%27SENTINEL-1%27)%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27platformSerialIdentifier%27%20and%20i0/Value%20eq%20%27A%27))%20or%20(Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27platformSerialIdentifier%27%20and%20i0/Value%20eq%20%27B%27))))%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27productType%27%20and%20i0/Value%20eq%20%27GRD%27))))%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27operationalMode%27%20and%20i0/Value%20eq%20%27IW%27))))%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27orbitDirection%27%20and%20i0/Value%20eq%20%27ASCENDING%27))%20or%20(Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27orbitDirection%27%20and%20i0/Value%20eq%20%27DESCENDING%27))))%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27polarisationChannels%27%20and%20i0/Value%20eq%20%27VV%26VH%27)))))))))&$expand=Attributes&$expand=Assets&$orderby=ContentDate/Start%20asc&$top=20")

for year in range(2017, 2024+1):
    for month in range(1, 12+1):
        start_date = f'{year}-{month:02}-01T00:00:00.000Z'
        end_day = monthrange(year, month)[1]
        end_date = f'{year}-{month:02}-{end_day:02}T23:59:59.999Z'

        url_location = f"(Online eq true) and (OData.CSC.Intersects(Footprint=geography%27SRID=4326;POLYGON%20(({polygon}))%27)) and "

        url = (f"{url_start}"
               f"((ContentDate/Start ge {start_date} and ContentDate/Start le {end_date}) and "
               f"{url_location}"
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

        # Create monthly COGs
        create_monthly_cogs(outputs, year, month)

        break

