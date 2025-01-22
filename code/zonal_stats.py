# Import libraries
from osgeo import gdal
import pandas as pd
import time
from tqdm import tqdm
import sys
import geopandas as gpd
import matplotlib
matplotlib.use('nbAgg')
# from gdal_sheep import
# from save_xarray_to_gtiff import
from rasterstats import zonal_stats


def calc_zonal_stat(data_tif_path, shapefile_path):
    # Open the data in the tif file with gdal
    opn = gdal.Open(data_tif_path)
    # Count the bands or the number of layers in the tiff file (timesteps)
    bands = list(range(opn.RasterCount))

    # Create an empty list to store the zonal statistics data
    l = []
    # Loop over all the bands in the tif, extract the zonal stat of each band
    # then store it in a dictionary because there is more than 1 stat calculated per band
    for b in tqdm(bands):
        time.sleep(0.5)
        z = zonal_stats(shapefile_path, data_tif_path, stats="min mean max median", band=b + 1)
        dict_ = z[0]
        l.append(dict_)

    # the length of the nested list should be the same as the number of bands
    print(f'length of l is: {len(l)}')

    # Create a dataframe of the zonal stat
    lst_min = []
    lst_max = []
    lst_mean = []
    lst_median = []
    for i in range(len(l)):
        lst_min.append(l[i]['min'])
        lst_max.append(l[i]['max'])
        lst_mean.append(l[i]['mean'])
        lst_median.append(l[i]['median'])

    df = pd.DataFrame({"min": lst_min, "max": lst_max,
                       "mean": lst_mean, "median": lst_median})

    print(f'dataframe of zonal statistics is: {df}')

    # save the raw zonal stat in a pickle file
    pickle_path = f'/wp_data/sites/HatfieldThorne/MODIS/timeSeries/ZonalStats/{nm_dp_v}_{nm_lyr}_{start_dt}-{end_dt}_zonalstat_{nm_shp}_initial'
    df.to_pickle(pickle_path)

    return opn, bands, pickle_path

# Input Variables
shapefile_path = "/workspace/WorldPeatland/sites/util_shapefiles/Hatfield_Moors_sinusoidal.shp"
data_tif_path = "/wp_data/sites/HatfieldThorne/MODIS/timeSeries/MCD15A3H.061._Lai_500m.linear.smoothn.0.5.descaled.tif"
# Name of the shapefile or area clipped
nm_shp = 'Hatfield'
# Name of the data product and version
nm_dp_v = 'MCD15A3H.061'
# Name of the layer
nm_lyr = '_Lai_500m'
# Start date YMD
start_dt = '20130601'
# End dat YMD
end_dt = '20230630'


def main(site_directory, shapefile_path):

    opn, bands, pickle_path = calc_zonal_stat(data_tif_path, shapefile_path)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <site_root_data_dir> <shapefile_path>")  # the user has to input one argument
    else:
        site_directory = sys.argv[1]
        shapefile_path = sys.argv[2]
        main(site_directory, shapefile_path)