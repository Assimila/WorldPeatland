import yaml
from datetime import datetime, timedelta
import calendar
from tqdm import *
from osgeo import ogr
import shutil
import glob
import logging
from urllib.request import urlretrieve
import os
import json

import sys
sys.path.append('/home/ysarrouh/TATSSI/')

# TATSSI command line equivalent
from TATSSI.download.modis_downloader import get_modis_data

# Sentinel Downloaders
sys.path.insert(0,'/home/ysarrouh')
from SentinelDownloader import *

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def get_polygon(geojson_path):
    '''
    get_polygon function extract bbox layer of a GeoJson file site and gives an osgeo geometry object 
    
    INPUT
        - geojson_path (str) - path/ location of the geoJson file of the site (given by user)
    
    OUTPUT
        - polygon (osgeo.ogr.Geometry) - polygon geometry object containing extent lat and lon of the site 
        - site_name (str) - string of the site_area name from GeoJson file 
    '''
   
    # Open the GeoJSON file
    driver = ogr.GetDriverByName("GeoJSON")
    src_GeoJSON = driver.Open(geojson_path)

    if src_GeoJSON is None:
        raise Exception("Error opening GeoJSON file")
    
    try:
        # Get the layer from the GeoJSON file
        site_layer = src_GeoJSON.GetLayer()
        
        # Check if the geometry type is polygon
        if site_layer.GetGeomType() != ogr.wkbPolygon: # ogr.wkbPolygon = 3
            raise Exception("The GeoJSON geometry is not a polygon")

        # Get name of the geoJson area 
        feat = site_layer.GetFeature(0)
        site_area = feat.GetField(0) # site_area name 
        country = feat.GetField(1) # country name

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
        polygon.AddGeometry(ring) # Geometry object can perform directly intersection on it

        # Close the GeoJSON file
        src_GeoJSON = None

        return polygon, site_area, country
    
    except Exception as e:
        src_GeoJSON = None
        raise e


def create_dir(output_dir, directory):
    
    '''
    create_dir fucntion will first check if the directory already exist if not it will 
    create a directory where it will store the data to be downloaded
    
    INPUTS:
        - output_dir (str/path) - specified by the user where they want the data to be downloaded
        - directory (str) - specified by each step in the code to create, usually its the name of the data product to be downloaded
    '''

    # Path 
    path = os.path.join(output_dir, directory) 
    
    if not os.path.exists(path):
        os.makedirs(path)
        LOG.info(f"Directory '{path}' created successfully.")
    else:
        LOG.info(f"Directory '{path}' already exists.")

    return path

def ogrIntersection(tiles_layer, site_bbox):

    ''' 
    ogrIntersection function finds the MODIS tile corresponding to the shapefile location 
    
    INPUTS:
        - tiles_layer - in this example its the MODIS sinusoidal world grid file
        - site_bbox - the bbox of the site to be matched
        
    OUTPUTS:
        - tiles (list) - intersection information, in this case the corresponding MODIS tile h and v value 
    '''
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
    
    '''
    
    change the string format to be able to run the get_modis_data
    
    INPUT format: h:19 v:4 => OUTPUT format: h19v04
    
    '''
    
    # Split the input string where there is blank space
    parts = input_str.split() # parts is now a list of 2 strings 

    # Process each part to remove ':' and leading zeros
    formatted_parts = []
    # loop over both parts of the plitted string
    for part in parts:
        key, value = part.split(':') # split the parts seperated by :
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

    data[0]['site_area'] = site_area # add site_area name
    data[0]['country'] = country # add country name to get the VIIRS S-NPP
    data[0]['tiles']= tiles # add tile 
    data[0]['bbox'] = polygon.ExportToWkt() #poly string of coordinates

    with open(dst_config, "w") as f:
        data = yaml.dump(
        data, stream=f, default_flow_style=False, sort_keys=False)
        
    LOG.info(f'Config file has been created and saved here {dst_config}')
    
def get_modis_timestep(path_modis, start_date, end_date, tiles, n_threads, _username, _password ):
    
    '''get_modis_timestep function will download MODIS data for albedo MCD43A3.61 with 8 days time step
    there is no need for now to download the daily data'''
        
    delta = timedelta(days=8)

    # Initialize a list with the first date also as datetime
    l = [start_date]
    while start_date <= end_date:
        start_date += delta
        l.append(start_date)
    # Get the data
    for t in l:
        print (t)
        get_modis_data('MOTA', 'MCD43A3.061', format_string(tiles[0]), 
                       path_modis + '/MCD43A3.061/', t,
                       t, n_threads, _username, _password)  
    
    
def get_modis_downloader(products, start_date, end_date, site_directory, site_area, format_tiles):
    
    '''this function will loop over the data products to be downloaded and choose accordingly which way to download 
    the data
    for albedo we do not need daily data, for now looping seperatly over 8 days timedelta'''

    # create a subdirectory in the site folder to store sentinel data
    path_modis = create_dir(site_directory, 'MODIS')
    
    n_threads = 6
    _username, _password = read_config_cred()
    
    for i in tqdm(range(len(products))):
        
        '''loop over all the MODIS product to be downloaded'''
                   
        if products[i]['product'] == 'MCD43A3.061':

            get_modis_timestep (products, start_date, end_date, site_directory, site_area, format_tiles)

            continue

        else: 

            print(products[i]['product'])
            print(f'start_date:{start_date}')
            print(f'end_date:{end_date}')

            # Set the date strings as datetime.datetime so that get_modis_data works 
            get_modis_data(products[i]['platform'], products[i]['product'], format_tiles, 
                           path_modis + f"/{products[i]['product']}/", 
                           start_date, end_date, n_threads, _username, _password)

        LOG.info(f"MODIS {products[i]['product']} download complete for {site_area}-{format_tiles}")
        
def sentinel_file_checker(path_sentinel, site_area):
    
    '''
    
    This function will extract the month and the year of the sentinel files already downloaded
    this only checks the directory set for the site_area. It also checks for empty files, if file
    is empty remove/ delete the file and do not include the date in the list so that the month is
    downloaded again. It is important to keep the same name for a same geographical area or site as
    the directory would change, thus, the checker would not be able to look for the correct files. 
    This checker for now checks files only on VH_ASCENDING
    
    INPUTS
        - path_sentinel (str) - its the path created for the sentinel data of this sepecific site_area
        - site_area (str) - site_name obtained from the geojson file set by user
    OUTPUTS
        - dt_l (list of datetime.date) - list of the month and dates of the already downloaded sentinel files 
            that the checker has found in this directory. 
    '''
    
    # TODO for now only checking for VH_asc might need to check all other bands?    
    l = sorted(glob.glob(path_sentinel+f'/datacube/S1_GRD/VH_ASCENDING/{site_area}/*'))
    
    dt_l = []
    for f in range(len(l)):

        # check if file is empty
        if os.path.getsize(l[f]) == 0:
            # if true remove the file
            os.remove(l[f])
            continue # skip all the rest so that the corrupted file is downloaded again
        else: 
            base_name = os.path.splitext(os.path.basename(l[f]))[0]
            # Split the file name by underscores
            parts = base_name.split('_')
            dt = parts[-1]+'-01' # last part is the date and add first of the month 

            dt = datetime.strptime(dt, '%Y-%m-%d').date()
            dt_l.append(dt)

    return dt_l

def generate_dates(start_date, end_date):
    
    ''' 
    generate_dates will generate the first day of the start month and then every first day of 
    the month until reaching the end date 
    
    INPUTS
        - start_date (datetime.date) - set by user
        - end_date (datetime.date) - set by user
        
    '''
    
    year = start_date.year
    month = start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        yield datetime(year, month, 1).date()
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
            
def get_sentinel(start_date, end_date, site_area, site_directory, geojson_path):
    
    # create a subdirectory in the site folder to store sentinel data
    path_sentinel = create_dir(site_directory, 'Sentinel')
    
    # get the list of dates already downloaded 
    dt_l = sentinel_file_checker(path_sentinel, site_area)
    
    if dt_l != 0:
        
        # if dt_l is not empty and data is already downloaded then 
        # download the data by monthly chunks
    
        # run the dates generator and put it in a list as a calendar reference
        dates_list = list(generate_dates(start_date.date(), end_date.date()))
                          
        for i, j in enumerate(dates_list):

            if j in dt_l:
                LOG.info(f'Sentinel data for {site_area}_{j} already downloaded')
            
            else:
                # Find the end of the month
                j_end = (j.year, j.month, calendar.monthrange(j.year, j.month)[1])

                # run sentinel downloaders per month 
                sd = SentinelDownloader(geojson_path, j, j_end)
                new_dwn_files = sd.download_raw_all(path_sentinel +'/rawdata/', manual_key = site_area)
                sd.write_raw_files_to_datacube(new_dwn_files, path_sentinel + '/datacube/')    
                              
    else: 
        LOG.info(f'Starting to download Sentinel data for {site_area}')

        # 1. Call SentinelDownloader class 
        sd = SentinelDownloader(geojson_path, start_date.date(),
                                end_date.date())


        # 2. download all raw data in the rawdata folder 
        new_dwn_files = sd.download_raw_all(path_sentinel +'/rawdata/', manual_key = site_area)

        LOG.info(f"Sentinel raw data for {site_area} download complete in this directory {path_sentinel +'/rawdata/'}")

        # 3. write the files into the datacube structure as tiffs
        sd.write_raw_files_to_datacube(new_dwn_files, path_sentinel + '/datacube/')

        LOG.info(f"Sentinel data for {site_area} added to the datacube {path_sentinel +'/datacube/'}")
        
    # remove all rawdata after datacube created
    shutil.rmtree(path_sentinel + '/rawdata/')

def get_viirs_archive(start_date, country, site_area, site_directory):
    
    '''get_viirs_archive function gets the viirs_snpp (sp) data from 2012 till 2021 only'''
    
    # create a subdirectory in the site folder to store VIIRS data
    path_viirs = create_dir(site_directory, 'VIIRS')
    
    for y in range(datetime.strptime(start_date, '%Y-%m-%d').year, 2021+1):

        url = (f"https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/"
            f"viirs-snpp_{y}_{country}.csv")
        filename = path_viirs + f'/viirs-snpp_{y}_{country}.csv'

        urlretrieve(url, filename)   
        
    LOG.info(f'VIIRS SNPP data for {site_area} added to {filename}')
    

def read_config_cred():
    """
    Read downloaders config file
    """

    fname = os.path.join('/home/ysarrouh/WorldPeatlands/', 'config_cred.json')
    with open(fname) as f:
        credentials = json.load(f)
        
    username = credentials['username']
    password = credentials['password']

    return username, password


def main(geojson_path, output_dir):
    
    '''
    
    INPUTs:
    - geojson_path is the path of the json site it needs to contain at least name of the site 
    and country where the site is located
    - output_dir where the user wants to the data to be downloaded 
    
    '''
    
    # check if GeoJson file exists    
    if not os.path.isfile(geojson_path):
        LOG.error('GeoJSON file does not exist')
        return 
    
    if not os.path.exists(output_dir):
        LOG.error(f'Output director does not exist: {output_dir}')
        return
    
    try: 
        # get info from Json file 
        polygon, site_area, country = get_polygon(geojson_path)
        
    except Exception as e:
        LOG.error(str(e))
        return
    
    # inform user processing started
    LOG.info(f'{site_area} is now processing')
    
    # Create a site specific directory   
    site_directory = create_dir(output_dir, site_area) # output_dir set by user
    
    # Read MODIS tiles KML as layer 
    fname = '/home/glopez/Projects/TATSSI/data/kmz/modis_sin.kml'
    driver  = ogr.GetDriverByName('KML')
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
    config_src = '/home/ysarrouh/WorldPeatlands/template_config.yml'
    dst_config = site_directory + f'/{site_area}_config.yml'
    
    shutil.copyfile(config_src, dst_config)
    
    # read config 
    start_date, end_date, products = read_config(dst_config)
    
    # update the new config created with the data from geojson 
    update_config(dst_config, site_area, format_tiles, polygon, country)

    LOG.info(f'Starting to download MODIS data for {site_area}')
    
    # set date strings as datetime objects both get_modis and get_sentinel will need it as datetime object
    _start_date = datetime.strptime(start_date, '%Y-%m-%d')
    _end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    get_modis_downloader(products, _start_date, _end_date, site_directory, site_area, format_tiles)

    LOG.info(f'MODIS data download completed for {site_area}')  
    
    
    get_sentinel(_start_date, _end_date, site_area, site_directory, geojson_path)
    
    get_viirs_archive(start_date, country, site_area, site_directory)


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <geojson_path> <output_dir>") # the user has to input two arguments  
    else:
        geojson_path = sys.argv[1] # location of the second item in the list which is the first argument geojson site location 
        output_dir = sys.argv[2] # location of output downloaded data 3rd item in the list which is the 2nd argument
        main(geojson_path, output_dir)
        
