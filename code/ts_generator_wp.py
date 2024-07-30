
from downloader_wp_test import *
from TATSSI.TATSSI.time_series.generator import Generator

# use the sys path only if working in jupyter notebook
# import sys
# sys.path.append('/workspace/TATSSI')
# sys.path.insert(0, '/workspace/WorldPeatland/code/')

'''ts_generator_wp 2nd script to run, it will generate time series for the MODIS data'''


# need to update TATSSI/TATSSI/qa/EOS catalogue from APPEARS to contain all the updated products and version 
# we can do that by removing the current pkl files in EOS and running TATSSI UI in the download data tab 
# this will initiate by itself getting all the available pkl files in APPEARS


# TODO check for existing MODIS time series files 
# it seems to work only in the UI and not in the scripts

def get_ts(site_directory, json_path):
    """
    INPUT site_directory from user, path to the folder where downloader_wp was run to download all 
    products 
    """
    modis_dir = site_directory + 'MODIS/'
    print('modis directory', modis_dir)

    if not os.path.exists(site_directory):
        LOG.error(f'This director does not exist: {site_directory}')
        return

    # check if a MODIS file exists in this directory 
    if not os.path.exists(modis_dir):
        LOG.error(f'A MODIS file does not exist: {modis_dir}')
        return

    # Getting the list of directories 
    dir_ = os.listdir(modis_dir)

    # Checking if the list is empty or not 
    if len(dir_) == 0:
        print(f"This directory {modis_dir} is empty")
        return

        # Open the GeoJSON file
    driver = ogr.GetDriverByName("GeoJSON")
    src_json = driver.Open(json_path)

    # Get the layer from the GeoJSON file
    site_layer = src_json.GetLayer()

    # Extract the bounding box coordinates
    min_x, max_x, min_y, max_y = site_layer.GetExtent()

    source_srs = osr.SpatialReference()
    source_srs.ImportFromProj4('+proj=longlat +datum=WGS84 +no_defs +type=crs')  # WGS84

    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs')

    # Create a coordinate transformation
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Create points for each corner of the bounding box
    bottom_right = ogr.Geometry(ogr.wkbPoint)
    bottom_right.AddPoint(max_x, min_y)

    top_left = ogr.Geometry(ogr.wkbPoint)
    top_left.AddPoint(min_x, max_y)

    # Transform the points
    bottom_right.Transform(transform)
    top_left.Transform(transform)

    # Extract the transformed coordinates
    min_x_sinusoidal, max_y_sinusoidal = top_left.GetX(), top_left.GetY()
    max_x_sinusoidal, min_y_sinusoidal = bottom_right.GetX(), bottom_right.GetY()

    product_dir = [item for item in dir_ if item.startswith('M')]

    for i, n in enumerate(product_dir):

        LOG.info(n)
        product_name, version = n.rsplit('.', 1)

        # since we want to do the processing per tile 
        # glob will get all the tile files for the product we are looping in 
        output_dirs = glob.glob(modis_dir + n + f'/*/')

        for output_dir in output_dirs:

            print(output_dir)

            # Create for MCD64A1 because the files are in hdf format just
            # for the processing to be able to use the data do not apply mask 
            # later on MCD64A1 with the apply_qa.py

            if os.path.getsize(output_dir) == 0:
                LOG.error(f'{output_dir} this file is empty')
                continue

            # minX, maxX , minY, maxY should be in Sinusoidal and the above order
            extent = (min_x_sinusoidal, max_x_sinusoidal, min_y_sinusoidal, max_y_sinusoidal)

            # TATSSI Time Series Generator object
            tsg = Generator(source_dir=output_dir,
                            product=product_name,
                            version=version,
                            data_format='hdf',
                            progressBar=None,
                            preprocessed=True,
                            extent=extent)

            LOG.info('++++++++++++++++++++++++++++++++++++++++++')
            LOG.info(output_dir)
            LOG.info(product_name)
            LOG.info(version)

            tsg.generate_time_series()

            LOG.info(f'time series generated for this product {n}')


def main(site_directory, json_path):
    """
    INPUT
        site_directory - str - path to a specific site where all data were previously downloaded
    """
    get_ts(site_directory, json_path)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <site_directory>")  # the user has to input one argument
    else:
        site_directory = sys.argv[1]
        json_path = sys.argv[2]
        main(site_directory, json_path)

# example how to run this script in command line (if script run in jupyter notebook)
# you should be where the code is saved in /workspace/WorldPeatland/code
# python ts_generator_wp.py /wp_data/sites/Degero/ /workspace/WorldPeatland/sites/Degero.geojson

# if you are using PyCharm, you should be in /workspace and run the following command
# python -m WorldPeatland.code.ts_generator_wp /wp_data/sites/Norfolk/ /workspace/WorldPeatland/sites/Norfolk.geojson
