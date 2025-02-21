import json
import logging
import shutil
import os
from osgeo import ogr
import subprocess
import yaml
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
from TATSSI.TATSSI.download.modis_downloader import get_modis_data
from WorldPeatland.settings import ROOT_DATA_DIR, EARTH_DATA_CRED
from WorldPeatland.code.utils import create_dir

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

'''downloader_wp first script to run: it will download all MODIS data for the input geojson file'''


def get_polygon(geojson_path):
    """
    get_polygon function extract bbox layer of a GeoJson file site and gives an osgeo geometry object 
    
    INPUT
        - geojson_path (str) - path/ location of the geoJson file of the site (given by user)
    
    OUTPUT
        - polygon (osgeo.ogr.Geometry) - polygon geometry object containing extent lat and lon of the site 
        - site_name (str) - string of the site_area name from GeoJson file 
    """

    # Open the GeoJSON file
    driver = ogr.GetDriverByName("GeoJSON")
    src_GeoJSON = driver.Open(geojson_path)

    if src_GeoJSON is None:
        raise Exception("Error opening GeoJSON file")

    try:
        # Get the layer from the GeoJSON file
        site_layer = src_GeoJSON.GetLayer()

        # Check if the geometry type is polygon
        if site_layer.GetGeomType() != ogr.wkbPolygon:  # ogr.wkbPolygon = 3
            raise Exception("The GeoJSON geometry is not a polygon")

        # Get name of the geoJson area 
        feat = site_layer.GetFeature(0)
        site_area = feat.GetField("site_area")  # site_area name
        country = feat.GetField("country")  # country name

        # Get the extent (bounding box) of the layer
        extent = site_layer.GetExtent()

        # Extract the bounding box coordinates
        min_x, max_x, min_y, max_y = extent

        # Create a polygon geometry from the bounding box coordinates
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(min_x, min_y)
        ring.AddPoint(max_x, min_y)
        ring.AddPoint(max_x, max_y)
        ring.AddPoint(min_x, max_y)
        ring.AddPoint(min_x, min_y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)  # Geometry object can perform directly intersection on it

        # Close the GeoJSON file
        src_GeoJSON = None

        return polygon, site_area, country

    except Exception as e:
        src_GeoJSON = None
        raise e


def ogrIntersection(tiles_layer, site_bbox):
    """
    ogrIntersection function finds the MODIS tile corresponding to the shapefile location

    INPUTS:
        - tiles_layer - in this example it's the MODIS sinusoidal world grid file
        - site_bbox - the bbox of the site to be matched

    OUTPUTS:
        - tiles (list) - intersection information, in this case the corresponding MODIS tile h and v value
    """

    # List with tiles for every feature in site GeoJSON 
    # it has to be a list of strings to include the bbox that might intersect more than 1 MODIS tile 
    tiles = []

    # Find overlapping features
    for feat1 in tiles_layer:
        geom1 = feat1.GetGeometryRef()

        if site_bbox.Intersects(geom1):
            # Get field 0 containing the tile index
            tiles.append(feat1.GetField(0))

    return tiles


def read_config(config_fname):
    """
    Read downloaders config file - Gerardo Saldana
    """
    with open(config_fname) as f:
        data = yaml.full_load(f)

    # Information from the first list index 0 about the site
    start_date = data[0]['start_date']
    end_date = data[0]['end_date']

    # Information about the first EO data product to download list index 1 
    products = data[1]['products']
    return start_date, end_date, products


def format_string(input_str):
    """
    Change the string format to be able to run the get_modis_data

    INPUT format: h:19 v:4 => OUTPUT format: h19v04
    """

    # Split the input string where there is blank space
    parts = input_str.split()  # parts is now a list of 2 strings

    # Process each part to remove ':' and leading zeros
    formatted_parts = []
    # loop over both parts of the split string
    for part in parts:
        key, value = part.split(':')  # split the parts separated by :
        formatted_parts.append(f"{key}{int(value):02d}")
        # format specifier: the 'value' should be made up of 2 decimal characters (2d)
        # and 0 for if the value is less than 2 characters it will add 0 to fill the requirement

    # Join the formatted parts together
    formatted_string = ''.join(formatted_parts)

    return formatted_string


def update_config(dst_config, site_area, tiles, polygon, country):
    # Add the tiles name into the config file specific to this site
    with open(dst_config) as f:
        data = yaml.full_load(f)

    data[0]['site_area'] = site_area  # add site_area name
    data[0]['country'] = country  # add country name to get the VIIRS S-NPP
    data[0]['tiles'] = tiles  # add MODIS tile
    data[0]['bbox'] = polygon.ExportToWkt()  # poly string of coordinates

    with open(dst_config, "w") as f:
        data = yaml.dump(
            data, stream=f, default_flow_style=False, sort_keys=False)

    LOG.info(f'Config file has been created and saved here {dst_config}')


def get_modis_timestep(product, platform, path, start_date, end_date, format_tiles):
    """
    get_modis_timestep function will download MODIS data for albedo MCD43A3.61 and MCD43A2.061 with 8 days time step
    there is no need for now to download the daily data
    """

    n_threads = 6

    # Create the list of doy for which to order the image
    interval = 8
    doy_list = list(range(1, 365 + 1, interval))

    # Initialize the list with the start date
    date_list = [start_date]

    # Go through dates from start_date to end_date, checking for DOY matches
    current_date = start_date
    while current_date <= end_date:
        # Get the DOY of the current date
        doy = current_date.timetuple().tm_yday

        # If the DOY is in our interval list, add it to date_list
        if doy in doy_list:
            date_list.append(current_date)

        # Move to the next day
        current_date += timedelta(days=1)

    # Remove the duplicate start date if it's added
    date_list = list(dict.fromkeys(date_list))

    # Get the data
    for t in date_list:
        LOG.info(t)
        get_modis_data(platform, product, format_tiles, path, EARTH_DATA_CRED, t, t, n_threads)
        get_modis_data(platform, product, format_tiles, path, EARTH_DATA_CRED, t, t, n_threads)


def run_command(cmd: str):
    """
    Executes a command in the OS shell
    :param cmd: command to execute
    :return N/A:
    :raise Exception: If command executions fails
    """
    # initialize an error msg so if there is no error it will assign a none value to err_msg
    err_msg = None

    status = subprocess.call([cmd], shell=True)

    if status != 0:
        err_msg = f"{cmd} \n Failed"
        raise Exception(err_msg)

    return err_msg


def get_modis_downloader(products, start_date, end_date, path_modis, site_directory, site_area, format_tiles):
    """
    This function will loop over the data products to be downloaded and choose accordingly which way to download
    the data for albedo we do not need daily data, for now looping separately over 8 days timedelta
    """

    # create a subdirectory in the site folder to store modis link data
    path_site_modis = create_dir(site_directory, 'MODIS')

    n_threads = 6
    for i in tqdm(range(len(products))):

        '''loop over all the MODIS product to be downloaded'''

        path_product = create_dir(path_modis, products[i]['product'])
        product = products[i]['product']
        if product in ('MCD43A3.061', 'MCD43A2.061'):

            # loop over all list of tiles to be able to create a file for each tile
            for tile in format_tiles:
                path_tile = create_dir(path_product, tile)
                platform = products[i]['platform']
                get_modis_timestep(product, platform, path_tile, start_date, end_date, tile)

                path_site_product = create_dir(path_site_modis, f"{products[i]['product']}/{tile}")

                try:
                    # create link to the site modis tile related data
                    err_msg = run_command(f'ln -s {path_tile + "/*hdf"} {path_site_product}')

                    if err_msg:
                        raise Exception(f'Link already exists: {path_site_product}')
                except Exception as e:
                    LOG.error(e)
        else:

            for tile in format_tiles:

                # where the data will be downloaded
                path_tile = create_dir(path_product, tile)

                # Set the date strings as datetime.datetime so that get_modis_data works 
                get_modis_data(products[i]['platform'], products[i]['product'], tile, path_tile, EARTH_DATA_CRED,
                               start_date, end_date, n_threads)

                # where the data will be linked, this is in the site specific modis file 
                path_site_product = create_dir(path_site_modis, f"{products[i]['product']}/{tile}")

                try:
                    err_msg = run_command(f'ln -s {path_tile + "/*hdf"} {path_site_product}')

                    if err_msg:
                        raise Exception(f'Link already exists: {path_site_product}')
                except Exception as e:
                    LOG.error(e)

                LOG.info(f"MODIS {products[i]['product']} download complete for {site_area}-{format_tiles}")


def generate_dates(start_date, end_date):
    """
    Generate_dates will generate the first day of the start month and then every first day of
    the month until reaching the end date (exactly like the calendar, this list will be used
    as a reference)

    INPUTS
        - start_date (datetime.date) - set by user
        - end_date (datetime.date) - set by user
    """

    year = start_date.year
    month = start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        yield datetime(year, month, 1).date()
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def check_dates(start_date, end_date, sentinel_start_date):
    # for s1 downloaders
    if start_date and end_date < sentinel_start_date:
        LOG.info(f'No Sentinel 1 data available between these dates {start_date} and {end_date}')
        start_date = 0
        end_date = 0
    elif start_date < sentinel_start_date and end_date > sentinel_start_date:
        # set the _start_date to be equal to the sentinel data availability
        LOG.info(f'Sentinel data is only available after {sentinel_start_date}')
        start_date = sentinel_start_date
    return start_date, end_date


def main(GEOJSON_PATH):
    # check if GeoJson file exists    
    if not os.path.isfile(GEOJSON_PATH):
        LOG.error('GeoJSON file path set in settings.py does not exist, update settings.py')
        return

    if not os.path.exists(ROOT_DATA_DIR):
        LOG.error(f"ROOT_DATA_DIR that you've set in the settings.py does not exist: {ROOT_DATA_DIR}  "
                  f"Please change the entry in your settings.py")
        return

    try:
        # get info from Json file 
        polygon, site_area, country = get_polygon(GEOJSON_PATH)

    except Exception as e:
        LOG.error(str(e))
        return

    # inform user processing started
    LOG.info(f'{site_area} is now processing')

    # Create a site specific directory   
    site_directory = create_dir(ROOT_DATA_DIR, site_area)  # output_dir set by user

    # Read MODIS tiles KML as layer
    path_downloader = os.path.abspath(__file__)
    fname = '../modis_tiles/modis_sin.kml'
    fname = os.path.normpath(os.path.join(os.path.dirname(path_downloader), fname))

    driver = ogr.GetDriverByName('KML')
    src_kml = driver.Open(fname)
    tiles_layer = src_kml.GetLayer()

    # get intersection tiles with site_area
    tiles = ogrIntersection(tiles_layer, polygon)

    format_tiles = []
    for i in range(len(tiles)):
        j = format_string(tiles[i])
        print(j)
        format_tiles.append(j)

    # create a copy of the template config file
    config_src = '../template_config.yml'
    config_src = os.path.normpath(os.path.join(os.path.dirname(path_downloader), config_src))
    dst_config = site_directory + f'/{site_area}_config.yml'

    shutil.copyfile(config_src, dst_config)

    # read config 
    start_date, end_date, products = read_config(dst_config)

    # update the new config created with the data from geojson 
    update_config(dst_config, site_area, format_tiles, polygon, country)

    # set date strings as datetime objects both get_modis and get_sentinel will need it as datetime object
    _start_date = datetime.strptime(start_date, '%Y-%m-%d')
    _end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # create a data/MODIS directory to store all products not in a site specific file
    # path_modis = create_dir(ROOT_DATA_DIR, 'raw_tiles/MODIS')
    # TODO change back to create_dir
    path_modis = '/data/MODIS'
    LOG.info(f'Starting to download MODIS data for {site_area}')
    get_modis_downloader(products, _start_date, _end_date, path_modis, site_directory, site_area, format_tiles)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <geojson_fname>")  # the user has to input 1 arguments
    else:
        # location of the second item in the list which is the first argument geojson site location
        GEOJSON_PATH = sys.argv[1]
        main(GEOJSON_PATH)



