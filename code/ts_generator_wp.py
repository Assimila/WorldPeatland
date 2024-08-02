
import numpy as np
from WorldPeatland.code.downloader_wp_test import *
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

def get_extent(fname):
    """
    Get extent from GeoJSON file as:
    xmin, xmax, ymin, ymax
    """
    d = ogr.Open(fname)
    a = d.GetLayer()
    return a.GetExtent()


def transform_bbox(bbox, edge_samples=11):
    """
    source of code
    https://gis.stackexchange.com/questions/165020/how-to-calculate-the-bounding-box-in-projected-coordinates
    """

    source_srs = osr.SpatialReference()
    source_srs.ImportFromProj4('+proj=longlat +datum=WGS84 +no_defs +type=crs')  # WGS84

    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs')

    # Create a coordinate transformation
    transformer = osr.CoordinateTransformation(source_srs, target_srs)

    p_0 = np.array((bbox[0], bbox[3]))
    p_1 = np.array((bbox[0], bbox[1]))
    p_2 = np.array((bbox[2], bbox[1]))
    p_3 = np.array((bbox[2], bbox[3]))

    def _transform_point(point):
        trans_x, trans_y, _ = (transformer.TransformPoint(*point))
        return trans_x, trans_y

    transformed_bbox = [
        bounding_fn(
            [_transform_point(
                p_a * v + p_b * (1-v)) for v in np.linspace(
                0, 1, edge_samples)])
        for p_a, p_b, bounding_fn in [
            (p_0, p_1, lambda p_list: min([p[0] for p in p_list])),
            (p_1, p_2, lambda p_list: min([p[1] for p in p_list])),
            (p_2, p_3, lambda p_list: max([p[0] for p in p_list])),
            (p_3, p_0, lambda p_list: max([p[1] for p in p_list]))]]

    return transformed_bbox


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

    # Extract the bounding box coordinates
    min_x, max_x, min_y, max_y = get_extent(json_path)

    # Create bbox to fit in the function
    bbox = [min_x, min_y, max_x, max_y]

    # transformed_bbox is in the same order as the input bbox
    # minX, minY, maxX, maxY
    transformed_bbox = transform_bbox(bbox, edge_samples=11)

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
            # reorder the transformed_bbox list
            extent = [transformed_bbox[0], transformed_bbox[2], transformed_bbox[1], transformed_bbox[3]]

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
