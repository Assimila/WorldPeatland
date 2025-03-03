# Import libraries
from osgeo import gdal
import pandas as pd
import time
from tqdm import tqdm
from glob import glob
import sys
import os
from datetime import datetime
import logging
import matplotlib

matplotlib.use('nbAgg')
from rasterstats import zonal_stats
from WorldPeatland.code.download_modis import create_dir
from WorldPeatland.code.gdal_sheep import _get_FillValue

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def calc_zonal_stat(data_tif_path, shapefile_path):
    # Open the data in the tif file with gdal
    opn = gdal.Open(data_tif_path)
    # Count the bands or the number of layers in the tiff file (timesteps)
    bands = list(range(opn.RasterCount))

    # get fill value from opn
    _FillValue = _get_FillValue(opn)
    # Create an empty list to store the zonal statistics data
    l = []
    # Loop over all the bands in the tif, extract the zonal stat of each band
    # then store it in a dictionary because there is more than 1 stat calculated per band
    for b in tqdm(bands):
        time.sleep(0.5)
        z = zonal_stats(
            shapefile_path,
            data_tif_path,
            stats="min mean max median",
            band=b + 1,
            nodata=_FillValue
        )
        dict_ = z[0]
        l.append(dict_)

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

    LOG.info(f'ZonalStats successfully calculated for {data_tif_path}')
    return opn, bands, df


def get_dts(opn, bands):
    dts = []
    for i in range(len(bands)):
        lyr = opn.GetRasterBand(i + 1)
        mtd = lyr.GetMetadata()
        time = mtd['time']
        time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f000').strftime('%d-%m-%Y')
        dts.append(time)
    return dts


def main(site_directory, shapefile_path):
    data_tif_list = glob(site_directory + 'MODIS/timeSeries/*tif')
    modis_timeSeries_path = site_directory + f'/MODIS/timeSeries/'

    for data_tif_path in data_tif_list:
        # get tif file name
        tif_filename = os.path.basename(data_tif_path)
        LOG.info(f'ZonalStats calculation started for {tif_filename}')
        opn, bands, df = calc_zonal_stat(data_tif_path, shapefile_path)

        dts = get_dts(opn, bands)
        # set the dts as index of the dataframe
        df['Dates'] = dts
        df['Dates'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')
        df.set_index('Dates', inplace=True)

        # create pkl_filename from tif_filename
        pkl_filename = tif_filename.replace('.tif', '.zonalStats.pkl')
        # save the raw zonal stat in a pickle file
        output_path = create_dir(modis_timeSeries_path, 'ZonalStats')
        output_path = os.path.join(output_path, pkl_filename)
        df.to_pickle(output_path)
        LOG.info(f'Zonal stats successfully saved here: {output_path}')


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage: python script.py <site_root_data_dir> <shapefile_path>")  # the user has to input one argument
    else:
        site_directory = sys.argv[1]
        shapefile_path = sys.argv[2]
        main(site_directory, shapefile_path)
