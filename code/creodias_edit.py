import os
import requests
from glob import glob
from requests.utils import requote_uri
from osgeo import gdal, ogr
import json
from calendar import monthrange
import xml.etree.ElementTree as ET
import pickle
import sys


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


def add_mtd(OUTPUTDIR, year, month, S3Paths):

    # dataset geotiff path
    monthly_cog_dir = os.path.join(OUTPUTDIR, f'datacube/S2_SR/B02/S2_SR_B02_{year}-{month:02}.tif')
    ds = gdal.Open(monthly_cog_dir, gdal.GA_Update)
    for i in range(ds.RasterCount):
        # Raster band starts with 1 and not 0
        rb = ds.GetRasterBand(i+1)
        mtd = rb.GetMetadata()
        # get time component of the band
        basename = os.path.basename(S3Paths[i]) # list count starts with 0
        time = basename.split('_')[2]
        # get metadata, need the time aspect
        metadata = get_metadata_path(time, S3Paths)
        sza, saa, vza, vaa = get_angle(metadata)

        # Add to mtd current dictionary the angular information
        mtd['sza'] = sza
        mtd['saa'] = saa
        mtd['vza'] = vza
        mtd['vaa'] = vaa

        # Update the band metadata with the new angles
        rb.SetMetadata(mtd)


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

    # OUTPUTDIR = '/wp_data/sites/Degero/Sentinel/MSIL2A'
    OUTPUTDIR = os.path.join(OUTPUT_DIR, 'MSIL2A_test')
    create_dir(OUTPUTDIR)

    # Create a file to store the sensing dates pickle files
    OUTPUTDIR_sensing_dates = os.path.join(OUTPUT_DIR, 'sensing_dates')
    create_dir(OUTPUTDIR_sensing_dates)

    # geojson_fname = '/workspace/WorldPeatland/sites/Degero.geojson'
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

            # Set the in band metadata in the B02 files only
            add_mtd(OUTPUTDIR, year, month, S3Paths)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python script.py <geojson_fname>, <OUTPUT_DIR>")  # the user has to input two arguments
    else:
        # location of the second item in the list which is the first argument geojson site location
        geojson_fname = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]
        main(geojson_fname, OUTPUT_DIR)
# geojson_fname = '/workspace/WorldPeatland/sites/Degero.geojson'
# OUTPUT_DIR = '/wp_data/sites/Degero/Sentinel'