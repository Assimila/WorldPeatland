import re
import logging
import sys
import os
from osgeo import osr
import yaml

sys.path.insert(0, '/workspace/WorldPeatland/code/')

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def create_dir(output_dir, directory):
    """
    create_dir function will first check if the directory already exist if not it will
    create a directory where it will store the data to be downloaded

    INPUTS: - output_dir (str/path) - specified by the user where they want the data to be downloaded - directory (
    str) - specified by each step in the code to create, usually it's the name of the data product to be downloaded

    """

    # Join the path string
    path = os.path.join(output_dir, directory)  # directory should not contain '/' at the beginning of the file name
    # it will be considered as an absolute path and thus ignore output_dir

    if not os.path.exists(path):
        os.makedirs(path)
        LOG.info(f"Directory '{path}' created successfully.")
    else:
        LOG.info(f"Directory '{path}' already exists.")

    return path


def get_file_name(file_path):
    """
    get_file_name from the file path returns the modis data product name
    and the version

    INPUT
        - file_path (str) - it would be the one set by the user when running the downloader_wp
            + MODIS the path specific to download MODIS data
    OUTPUT
        - file_name[0] (str) - in this case it would be the MODIS data product name
        - file_name[1] (str) - in this case it would be the MODIS data product version
    """

    file_path_components = file_path.split('/')
    file_name = file_path_components[-1].rsplit('.', 1)
    return file_name[0], file_name[1]


def get_timestep_from_tif(tif):
    """
    Extract the timestep (date) from a geotif filename

    INPUT
        - tif (string) - tif file path where the filename ends with for example *2018-06.*extension*

    OUTPUT
        - timestep (string) - example 2018-06 (YYYY-MM)
    """

    # Search for the same file but for all other reflectance bands
    # get the date from tif file name
    filename = os.path.basename(tif)
    # Match using regex pattern in filename
    match = re.search(r'\d{4}-\d{2}', filename)  # date YYYY (4 digits) and MM (2 digits)

    return match.group()



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


def proj4_extract(opn):

    # get spatial reference from opn dataset
    proj_wkt = opn.GetProjection()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj_wkt)
    proj4_string = spatial_ref.ExportToProj4()

    return proj4_string