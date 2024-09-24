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
import xml.etree.ElementTree as ET
import pickle
import sys


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

    return proj_x, proj_y


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
        raise FileNotFoundError(f"Unable to open {img_path}")
    proj_wkt = opn.GetProjection()
    srs = osr.SpatialReference(wkt=proj_wkt)

    return srs


def reproject_srs(img_path, output_path, target_srs):
    opn = gdal.Open(img_path)

    if opn is None:
        raise FileNotFoundError(f"Unable to open {img_path}")

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
            reproj_img_path = reproj_img_path.replace(".vrt", "_reprojected.vrt")
            reproj_img_path = os.path.join(output_dir, 'VRTs', reproj_img_path)

            # run the reprojection function
            reproject_srs(img_path, reproj_img_path, reference_srs)
            reproj_inputs_dirs.append(reproj_img_path)

        else:
            reproj_inputs_dirs.append(img_path)

    return reproj_inputs_dirs


def create_subset(input_dirs, output_dir, extent, band):
    for i in range(len(input_dirs)):
        fname = glob(input_dirs[i])
        if len(fname) > 0:
            input_dirs[i] = fname[0]
        # TODO Check that files exist

    create_dir(output_dir)
    if band == 'MSK_CLDPRB_20m':
        _fname = os.path.basename(str(Path(input_dirs[0]).parent.parent.absolute()))
        output_fname = f'{_fname}_{band}'
    else:
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
    options = gdal.WarpOptions(format='VRT',
                               outputBounds=extent_native_crs)

    vrt = gdal.Warp(output_fname, input_dirs, options=options)

    vrt = None
    del vrt

    return output_fname


def get_crs(fname):
    """
    Get CRS in WKT string from a single file
    """
    d = gdal.Open(fname)
    proj = d.GetProjection()
    return proj


def create_daily_vrts(S3Paths, OUTPUTDIR, datasets, year, month, days, extent, product='S2_SR'):
    """
    Create mosaics for daily set of Sentinel-2 acquisitions
    """
    # Dictionary to store all ouputs per dataset-band
    outputs = {}

    # Find all images for a particular day
    for day in range(1, days + 1):
        date = f"{year}{month:02}{day:02}"
        images = []
        for img in S3Paths:
            if img.find(date) > 0:
                images.append(img)

        if len(images) == 0:
            images = []
            continue
        # If there is a single image just subset 
        for dataset in datasets:
            for band in datasets[dataset]:
                if band == 'MSK_CLDPRB_20m':
                    img_path = f'GRANULE/*/{dataset}/*{band}*.jp2'
                else:
                    img_path = f'GRANULE/*/*_DATA/{dataset}/*{band}*.jp2'
                output_dir = os.path.join(OUTPUTDIR, 'datacube',
                                          product, band, 'VRTs')

                images_path = []
                for i in range(len(images)):
                    images_path.append(os.path.join(images[i], img_path))

                ouput_fnames = create_subset(images_path,
                                             output_dir, extent, band)

                if band in outputs:
                    outputs[band].append(ouput_fnames)
                else:
                    outputs[band] = [ouput_fnames]
    return outputs


def get_angle(metadata):
    """
    get azimuth angle metadata from the MTD.xml file in SAFE
    
    INPUT
        - metadata - str of the MTD.xml file path 

    OUTPUT
        - azimuth_angle - str one value (per timestep)
    code reference:
    https://gis.stackexchange.com/questions/471487/get-mean-solar-azimuth-angle-from-sentinel-2-l2a-product
    """

    tree = ET.parse(metadata)
    root = tree.getroot()

    tile_angles_element = root.find('.//Tile_Angles')
    if tile_angles_element is not None:
        mean_sun_angle_element = tile_angles_element.find('.//Mean_Sun_Angle')
        if mean_sun_angle_element is not None:
            sza = mean_sun_angle_element.find('./ZENITH_ANGLE').text
            saa = mean_sun_angle_element.find('./AZIMUTH_ANGLE').text
        else:
            print("No Mean Sun Angle was found")
    # Iterate over the Mean_Viewing_Incidence_Angle elements to find bandId='2'
    for angle in root.findall('.//Mean_Viewing_Incidence_Angle'):
        if angle.attrib.get('bandId') == '2':
            vza = angle.find('ZENITH_ANGLE').text
            vaa = angle.find('AZIMUTH_ANGLE').text
            break

    return sza, saa, vza, vaa


def get_metadata_path(time, S3Paths):
    """
    get the MTD_xml file path from matching the time component 
    in the S3Paths file names
    """

    MTD_path = f'GRANULE/*/MTD_TL.xml'

    metadata = next((
        path for path in S3Paths if time in path),
        None)

    metadata = glob(os.path.join(metadata, MTD_path))
    return metadata[0]


def create_monthly_cogs(outputs, OUTPUTDIR, year, month, S3Paths, product='S2_SR'):
    """
    Create monthly DataCube COGs
    """
    for dataset in outputs:
        # Check srs of the time steps in a month
        output_dir = os.path.join(OUTPUTDIR, 'datacube', product, dataset)
        outputs[dataset] = check_srs(outputs[dataset], output_dir)
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
            if dataset == 'MSK_CLDPRB_20m':
                _time = _time.split('_')[3]
            else:
                _time = _time.split('_')[1]

            band = vrt.GetRasterBand(i + 1)

            if not dataset == 'MSK_CLDPRB_20m':

                metadata = get_metadata_path(_time, S3Paths)

                if metadata:

                    sza, saa, vza, vaa = get_angle(metadata)
                    band.SetMetadataItem('saa', saa)
                    band.SetMetadataItem('sza', sza)
                    band.SetMetadataItem('vza', vza)
                    band.SetMetadataItem('vaa', vaa)

                    metadata = None
                else:
                    print('azimuth angle not found')
            else:
                # If the dataset is MSK_CLDPRB ignore the angles because of different file paths
                # anw it is the same values throughout the bands no need to have it in here too
                continue

            _time = str(pd.to_datetime(_time, format='%Y%m%dT%H%M%S'))

            band.SetMetadataItem('add_offset', '0')
            band.SetMetadataItem('fill_value', '999')
            band.SetMetadataItem('product', product)
            band.SetMetadataItem('scale_factor', '1.0')
            band.SetMetadataItem('time', _time)
            band.SetMetadataItem('version', 'Sentinel-2_L2_Sen2Cor')

        del vrt

        translate_options = gdal.TranslateOptions(format='GTiff')

        # Get output dir from first VRT
        output_dir = Path(outputs[dataset][0])
        output_dir = str(output_dir.parent.parent.absolute())
        # Output file name
        output_cog_fname = f'{product}_{dataset}_{year}-{month:02}.tif'
        output_cog_fname = os.path.join(output_dir, output_cog_fname)

        tmp_ds = gdal.Translate(output_cog_fname, output_vrt_fname,
                                options=translate_options)

        # Clean tmp variables
        del tmp_ds
        del f


# =======================================================================#
# - Queries thi CREODIAS API to search data for specific year and month
# - Creates subset/mosaic for every day when acquisitions have been found
# - Creates monthly DQ COGs for every month
# =======================================================================#


def main(geojson_fname, OUTPUT_DIR):
    datasets = {'R10m': ['B02', 'B03', 'B04', 'B08'],
                'R20m': ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
                'R60m': ['B01'],
                'QI_DATA': ['MSK_CLDPRB_20m']}
    cloud_cover_le = 30

    #OUTPUTDIR = '/wp_data/sites/Degero/Sentinel/MSIL2A'
    OUTPUTDIR = os.path.join(OUTPUT_DIR, 'MSIL2A_test')
    create_dir(OUTPUTDIR)

    # Create a file to store the sensing dates pickle files
    OUTPUTDIR_sensing_dates = os.path.join(OUTPUT_DIR, 'sensing_dates')
    create_dir(OUTPUTDIR_sensing_dates)

    #geojson_fname = '/workspace/WorldPeatland/sites/Degero.geojson'
    extent = get_extent(geojson_fname)
    polygon = get_polygon(geojson_fname)

    url_start = (f"https://datahub.creodias.eu/odata/v1/Products?$filter="
                 f"((Attributes/OData.CSC.DoubleAttribute/any(i0:i0/Name eq %27cloudCover%27 "
                 f"and i0/Value le {cloud_cover_le})) and ")

    url_end = (f"(Online eq true) and "
               f"(OData.CSC.Intersects(Footprint=geography%27SRID=4326;POLYGON%20(("
               f"{polygon}"
               f"))%27)) and "
               f"(((((Collection/Name eq %27SENTINEL-2%27) and "
               f"(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name eq %27productType%27 "
               f"and i0/Value eq %27S2MSI2A%27)))) and "
               f"(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name eq %27processorVersion%27 "
               f"and i0/Value eq %2705.00%27)) or (Attributes/OData.CSC.StringAttribute/any(i0:i0/Name "
               f"eq %27processorVersion%27 and i0/Value eq %2705.09%27))))))))"
               f")&$expand=Attributes&$expand=Assets&$orderby=ContentDate/Start asc&$top=200")

    for year in range(2017, 2024 + 1):
        for month in range(1, 12 + 1):
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
            # set unique sensing dates or acquisition dates
            # these dates includes milliseconds, it would be impossible to have
            # the same full dates with different coverage over the site
            sensing_dates = []
            S3Paths = []
            for i, element in enumerate(response['value']):
                # Get acquisition date first date in the S3Path file name
                image_name = os.path.basename(element['S3Path'])
                sensing_date = image_name.split('_')[2]
                # Check if the date is not already there
                if sensing_date not in sensing_dates:
                    sensing_dates.append(sensing_date)
                    S3Paths.append(element['S3Path'])


                new_dir = os.path.join(OUTPUTDIR, image_name)
                try:
                    os.symlink(element['S3Path'], new_dir,
                               target_is_directory=True)

                except FileExistsError:
                    print(f"{element['S3Path']} already exists")
                    print()

            if len(S3Paths) > 0:

                # Save sensing dates as pickle
                pickle_fname = os.path.join(OUTPUTDIR_sensing_dates, f'{year}_{month}.pkl')
                # Open the file in binary write mode and save the set
                with open(pickle_fname, 'wb') as file:
                    pickle.dump(sensing_dates, file)
            else:
                continue


            # Create daily VRTs
            outputs = create_daily_vrts(S3Paths, OUTPUTDIR, datasets, year, month, end_day, extent)
            # Create monthly COGs
            create_monthly_cogs(outputs, OUTPUTDIR, year, month, S3Paths)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python script.py <geojson_fname>, <OUTPUT_DIR>")  # the user has to input two arguments
    else:
        # location of the second item in the list which is the first argument geojson site location
        geojson_fname = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]
        main(geojson_fname, OUTPUT_DIR)
#geojson_fname = '/workspace/WorldPeatland/sites/Degero.geojson'
#OUTPUT_DIR = '/wp_data/sites/Degero/Sentinel'