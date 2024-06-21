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
sys.path.append('/workspace/TATSSI/')

sys.path.insert(0,'/workspace/WorldPeatland/code/')
from downloader_wp_test import *

from TATSSI.time_series.generator import Generator  
    
    
def get_ts(modis_dir, json_path):


    print(modis_dir)
    print(json_path)
    product_name, version = 'MOD16A2GF', '061'

    # since we want to do the processing per tile 
    # glob will get all the tile files for the product we are looping in 
    output_dir = glob.glob(modis_dir + f'*/MOD16A2GF.061/*/')
    output_dir = output_dir[0]
    print(output_dir)

    # Open the GeoJSON file
    driver = ogr.GetDriverByName("GeoJSON")
    src_GeoJSON = driver.Open(json_path)
    
    # Get the layer from the GeoJSON file
    site_layer = src_GeoJSON.GetLayer()
    
    # Extract the bounding box coordinates
    min_x, max_x, min_y, max_y = site_layer.GetExtent() 

    source_srs = osr.SpatialReference()
    source_srs.ImportFromProj4('+proj=longlat +datum=WGS84 +no_defs +type=crs')  # WGS84

    target_srs = osr.SpatialReference()
    target_srs.ImportFromProj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs')

    # Create a coordinate transformation
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Create points for each corner of the bounding box
    bottom_left = ogr.Geometry(ogr.wkbPoint)
    bottom_left.AddPoint(min_x, min_y)

    top_right = ogr.Geometry(ogr.wkbPoint)
    top_right.AddPoint(max_x, max_y)

    # Transform the points
    bottom_left.Transform(transform)
    top_right.Transform(transform)    
    
    # Extract the transformed coordinates
    minX = bottom_left.GetX()
    maxX = top_right.GetX()
    minY = bottom_left.GetY()
    maxY = top_right.GetY()   

    # minX, maxX , minY, maxY should be in Sinusoidal and the above order
    extent = (minX, maxX , minY, maxY)

    # TATSSI Time Series Generator object
    tsg = Generator(source_dir=output_dir,
             product=product_name,
             version=version,
             data_format='hdf',
             progressBar=None,
             preprocessed=True,
             extent = extent)

    tsg.generate_time_series()
        
def main():

    jsons = glob.glob(f'/workspace/WorldPeatland/sites/*.geojson')
    jsons.sort()
    dirs = glob.glob(f'/wp_data/sites/*/')
    dirs.sort()

    d = dict(zip(dirs, jsons))

    for modis_dir, json_path in d.items():
        get_ts(modis_dir, json_path)
    
if __name__ == "__main__":

    if len(sys.argv) != 1:
        
         print("Usage: python ts_generator_single.py")
        
    else:
        main()



    

