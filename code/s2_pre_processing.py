import os.path
import yaml
import logging
import subprocess
from gdal_sheep import *
from MLEO_NN import *
from smoothn import smoothn
from save_xarray_to_gtiff_old import *
from datetime import datetime as dt
import re
import rasterio
import sys
sys.path.insert(0, '/workspace/WorldPeatland/code/')

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


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


def gdal_dt(e, time):
    '''
    gdal_dt function will open the tif file as an osegeo gdal dataset
    from gdal_sheep

    INPUTS:
        - e (str or tiff) - path the tiff file or the gdal dataset you want to
            open and save its datetimes
        - time (string) - check how the time variable is written in the tiff metadata
    Outputs:
        - arr (np.array) - return arr of the gdal dataset
        - dts (list) - list of the datetimes
        - saved_opn (osegeo gdal dataset) - saved dataset for its srs
    '''

    # Create an empty list to store the datatimes
    dts = []

    # Check if the input is a str which would be the tif file
    # otherwise it is already an opened gdal dataset
    if type(e) == str:
        # open the Dataset
        opn = gdal.Open(e)
    else:
        opn = e

    for i in range(1, opn.RasterCount + 1):
        rst = opn.GetRasterBand(i)
        meta = rst.GetMetadata()

        # following fill in with the corresponding format
        # 'time' check the metadata of the tiff to see what they call
        # could also be 'RANGEBEGINNINGDATE'
        # the time data
        x = meta[time]
        # also check the metadata to see how is the format of datetime data
        # dt_format = '%Y-%m-%dT%H:%M:%S.000000000'
        dt_format = '%Y-%m-%d %H:%M:%S'
        # dt_format = '%Y-%m-%d'
        t = dt.strptime(x, dt_format)

        # append it to the list
        dts.append(t)

    # save the last osegeodataset for its srs
    saved_opn = opn

    # open the array
    arr = opn.ReadAsArray()

    return arr, dts, saved_opn


def create_ds(regrid_dict, bands):
    
    """
    create_ds will generate an xarray ds of all the sentinel 2 datasets and cloudmask
    INPUTS:
        - bands (list) - keys of the dictionary with name of the bands
    OUTPUT:
        - ds (xarray.Dataset) - it contains all 15 sentinel bands plus the acm cloud mask 
            of all the files in the output_dir/Sentinel (total number of variables is 16)
    """

    stack, dts, opn = gdal_stack_dt(regrid_dict, 'time')

    stack_list = []
    for i in regrid_dict:
        stack, dts, opn = gdal_stack_dt(regrid_dict[i], 'time')
        stack_list.append(stack)
    stack_dict = dict(zip(bands, stack_list))

    xs, ys = create_coord_list(opn)

    ds = xr.Dataset(data_vars={i: (('time', 'latitude', 'longitude'), stack_dict[i])for i in bands},
                    coords={'time': dts, 'latitude': ys, 'longitude': xs})
    
    return ds, dts, ys, xs


def create_dir(output_dir, directory):
    
    """
    create_dir function will first check if the directory already exist if not it will
    create a directory where it will store the data to be downloaded
    
    INPUTS:
        - output_dir (str/path) - specified by the user where they want the data to be downloaded
        - directory (str) - specified by each step in the code to create,
        usually its the name of the data product to be downloaded
    """

    # Path 
    path = os.path.join(output_dir, directory) 
    
    if not os.path.exists(path):
        os.makedirs(path)
        LOG.info(f"Directory '{path}' created successfully.")
    else:
        LOG.info(f"Directory '{path}' already exists.")

    return path


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


def process_MLEONN(data, config_fname, dts, ys, xs, saved_path, tile_name, data_product):
    
    start_date, end_date, products = read_config(config_fname)
    
    # 1.Smooth data using smoothn can smoothn all the biophysical parameters
    # use dask concatenate
    smoothed_data = smoothn(y=data, s=10, isrobust=True, axis=0)[0]

    # 2.Create xarray to be able to interpolate in function of time 
    ds = xr.Dataset(data_vars={f'{data_product}_smooth': (('time', 'latitude', 'longitude'), smoothed_data)},
                    coords={'time': dts, 'latitude': ys, 'longitude': xs})
    
    # 3.Perform linear interpolation
    ds_linear = ds.interp(coords={'time': pa.date_range(start_date, end_date, freq='1D')}, method='linear')
    
    # 4.Set CRS attribute
    # get the proj4 str from the tif
    proj4_utm = '+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs'
    ds_linear.attrs['crs'] = proj4_utm
    
    # 5.Save as utm TIFF
    output_utm = os.path.join(saved_path, f'{data_product}_{tile_name}_smoothn_utm.tif')
    save_xarray_old(output_utm, ds_linear, f'{data_product}_smooth')

    # 6.If data is LAI, resample to 10 by 10 pixel size
    proj4_string = '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'
    # change projection from utm to sinusoidal 
    output_sinu = os.path.join(saved_path, f'{data_product}_{tile_name}_smoothn_sinusoidal_resampled.tif')
    ds = gdal.Open(output_utm)

    # reproject to sinusoidal and resample to 10 by 10 pixel size
    dsReprj = gdal.Warp(output_sinu, ds, dstSRS=proj4_string, xRes=10, yRes=10)
    ds = dsReprj = None  # close the files
    LOG.info(f'{data_product}_resampled and saved')
        
    # 8.Delete UTM files
    os.remove(output_utm)



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


def get_bands_SR_in_arrays(bd, tif, opn, month_tif, input_dict):
    """
    1. Get the surface reflectance bands (other than B2 SR)
    2. regrid to 10 m pixel resolution
    3. put the surface reflectance in arraqys + multiply by scaling factor
    4. add to input_dict

    OUTPUT
        - input_dict (dictionary) - populated input_dict with all surface reflectance bands needs for MLEONN
    """

    # get name of the spectral band from the tif path
    band_name = tif.split('/')[-2]

    # Check that all monthly tif files for all the reflectance have the same no. of timesteps as B02
    with rasterio.open(tif) as src:
        if src.count != opn.RasterCount:
            raise ValueError(f'Raster count for {tif} does not match B02 timesteps'
                             f'Expected: {opn.RasterCount}, Found: {src.count}')

    # regrid the tifs for the same B02 pixel size
    g = reproject_image(tif, month_tif)
    # g is a gdal.dataset, pick the same raster band number in the loop
    rb_g = g.GetRasterBand(bd)
    arr_g = rb_g.ReadAsArray()
    SR_BD_SCALE = 0.0001
    input_dict[f'{band_name}'] = (arr_g * SR_BD_SCALE).astype(np.float32)

    return input_dict


def prepare_run_MLEONN(bd, opn, month_tif, dts, reflectance_tifs, MLEONN_products_dict):
    """
    Prepare the input bands surface reflectance + Run MLEONN
    OUTPUT
        - MLEONN_products_dict (dictionary) - dict with keys as MLEONN products name and with
            corresponding values as list of arrays
    """
    # Start by adding the reflectance B02 array to the input_dictionary
    rb = opn.GetRasterBand(bd)
    # Read the band 2d array
    arr = rb.ReadAsArray()
    # multiply by scaling factor
    SR_BD_SCALE = 0.0001
    arr = (arr * SR_BD_SCALE).astype(np.float32)
    # create a dictionary where you will append all the 2d arrays needed as inputs for MLEONN
    input_dict = {'B02': arr}
    # Now loop over all other bands
    for tif in reflectance_tifs:
        # Populate the input_dict with descaled data from the other bands (other than B2)
        input_dict = get_bands_SR_in_arrays(bd, tif, opn, month_tif, input_dict)

    # Get the in band metadata information from the B02 monthly tif file
    mtd = rb.GetMetadata()  # mtd is a dictionary containing all in band metadata

    # Get the timestep from the metadata to reconstruct the monthly tif of the MLEONN products
    time_str = mtd['time']
    dt_format = '%Y-%m-%d %H:%M:%S'
    time = dt.strptime(time_str, dt_format)
    dts.append(time)

    # extract the angular information as numpy float 32 (it is set as str)
    vaa = np.float32(mtd['vaa'])
    vza = np.float32(mtd['vza'])
    saa = np.float32(mtd['saa'])
    sza = np.float32(mtd['sza'])

    # form a numpy array for each angle same shape as the B02 reflectance array
    vaa = np.full(arr.shape, vaa)
    vza = np.full(arr.shape, vza)
    saa = np.full(arr.shape, saa)
    sza = np.full(arr.shape, sza)

    # add the arrays to the input_dict
    input_dict['vaa'] = vaa
    input_dict['vza'] = vza
    input_dict['saa'] = saa
    input_dict['sza'] = sza

    # process MLEONN as one timestep per rasterband
    lai = LAI_evaluatePixelOrig(input_dict)
    fapar = FAPAR_evaluatePixelOrig(input_dict)
    fc = FC_evaluatePixel(input_dict)
    cab = CAB_evaluatePixel(input_dict)

    # Export the one timestep (one 2d array) of B8 and saa needed in for the dark pixels selection
    B8_arr = input_dict['B8']
    saa_arr = input_dict['saa']

    # Empty the input_dict
    input_dict = None

    # Append the MLEONN product arrays to the dictionary
    MLEONN_products_dict['lai'].append(lai)
    MLEONN_products_dict['fapar'].append(fapar)
    MLEONN_products_dict['fc'].append(fc)
    MLEONN_products_dict['cab'].append(cab)

    return MLEONN_products_dict, B8_arr, saa_arr
def create_monthly_cogs(product, S2_path, timestep, opn, dts, month_tif, MLEONN_products_dict):
    """
    Create cog monthly tiffs for the MLEONN products created
    1. stack the arrays all the days available with the month
    2. Create an xarray
    3. Get the crs proj-4 fromt he downloaded initial S2 B2 monthly tif
    4. save the xarray as tif
    """

    # loop over all MLEONN generated products to save them to cog monthly tiffs

    # stack the numpy arrays in the lists
    stacked_arr = np.stack(MLEONN_products_dict[f'{product}'], axis=0)

    # create name of the output directory for the MLEONN products
    output_dir = create_dir(os.path.dirname(S2_path), f'MLEONN/{product}')

    # Output file name
    output_cog_fname = f'{product}_{timestep}.tif'
    output_cog_fname = os.path.join(output_dir, output_cog_fname)

    # create an xarray for each MLEONN product
    xr = create_xarr(opn, product, stacked_arr, dts)

    # get crs or the proj4 str from the monthly-tif using the command line of gdalinfo
    command = ['gdalinfo', month_tif, '-proj4']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # use regex to find the string in the results
    proj4_match = re.search(r"PROJ\.4 string is:\n'(.*?)'", result.stdout)
    proj4_string = proj4_match.group(1)
    # set proj4 str as an attribute to the xarray so that it can be saved as a tiff
    xr.attrs['crs'] = proj4_string

    # edited version of save_xarray_to_gtiff
    save_tiff(output_cog_fname, xr, product)

    return output_cog_fname


def main(S2_path):
    
    # tile_name, _ = get_file_name(geojson_path)

    # Path to B02 datacube tiff files
    B02_path = os.path.join(S2_path, 'B2')

    # get B02 monthly tif files
    B02_tif_files_list= glob.glob(os.path.join(B02_path, '*.tif'))

    # Set reflectance bands list
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    for month_tif in B02_tif_files_list:

        # extract timestep from the filename
        timestep = get_timestep_from_tif(month_tif)
        reflectance_tifs = []
        for band in bands:
            search_pattern = os.path.join(S2_path, '**', band, f'*{timestep}*.tif')
            reflectance_tifs.extend(glob.glob(search_pattern, recursive=True))

        # Open the monthly B02 tif file
        opn = gdal.Open(month_tif)

        # Check that all monthly tif files for all the reflectance have the same no. of timesteps as B02
        for tif in reflectance_tifs:
            with rasterio.open(tif) as src:
                if src.count != opn.RasterCount:
                    raise ValueError(f'Raster count for {tif} does not match B02 timesteps'
                                     f'Expected: {opn.RasterCount}, Found: {src.count}')

        # MLEONN products name list
        MLEONN_products_name = ['lai', 'fapar', 'fc', 'cab']
        # Empty dictionary to store the biophysical MLEONN products
        MLEONN_products_dict = {}
        for product in MLEONN_products_name:
            MLEONN_products_dict[product] = []
        # Datetime empty list
        dts = []
        # Create an empty list to append the in month single days of B8 and saa
        B8_list = []
        saa_list = []

        # Now loop over each timestep in this month, loop over each raster in the tif
        # in this case bd as band meaning a 2d raster or one timestep and not reflectance bands
        for bd in range(opn.RasterCount):
            bd += 1  # gdal starts raster band count from 1
            ######### PREPARE INPUT BAND SR AND RUN MLEONN ###############
            MLEONN_products_dict, B8_arr, saa_arr = prepare_run_MLEONN(bd, opn, month_tif, dts, reflectance_tifs, MLEONN_products_dict)
            B8_list.append(B8_arr)
            saa_list.append(saa_arr)
        # this is descaled B8 array
        B8_arr = np.stack(B8_list, axis=0)
        saa_arr = np.stack(saa_list, axis=0)

        # empty dictionary  to store outputs file names
        outputs = {}
        for product in MLEONN_products_name:
            output_cog_fname = create_monthly_cogs(product, S2_path, timestep,
                                                   opn, dts, month_tif, MLEONN_products_dict)

            # If the product key is not already in the outputs dictionary, initialize it with an empty list
            if product not in outputs:
                outputs[product] = []
            # Append the output filename to the corresponding product key
            outputs[product].append(output_cog_fname)

       ################ Cloud bands ##############

        # Get the TOA cloud probability corresponding month cog
        # Search the TOA path
        toa_path = S2_path.replace('/MSIL2A/', '/MSIL1C/')
        toa_path = toa_path.replace('/S2_SR', '/S2_TOA/')
        search_pattern = os.path.join(toa_path, 'ACM', f'*{timestep}*.tif')
        # cprob has a 10m pixel resolution
        cprob_path = glob.glob(search_pattern, recursive=True)[0]
        cprob_arr, dts, saved_opn = gdal_dt(cprob_path, 'time')

        # Set a cloud probability threshold
        cld_prb_thresh = 0.5
        # apply the mask 1 for cprob >= 0.5 not cloudy is set as 0
        # pixel is cloud == 1
        is_cloud = np.where(cprob_arr >= cld_prb_thresh, 1, 0)

        ########## Shadow bands ##########

        # Create a non-water array based on SCL values (6 is water pixel)
        # Get the SCL corresponding month tif
        search_pattern = os.path.join(S2_path, '**', 'SCL', f'*{timestep}*.tif')
        SCL_path = glob.glob(search_pattern, recursive=True)[0]
        SCL_arr, dts, saved_opn = gdal_dt(SCL_path, 'time')
        # water class is 6 in SCL
        water_class = 6
        # if the pixel is water(=6) set the pixel as 1 (True)
        # if pixel not water then it is 0
        not_water = np.where(SCL_arr == water_class, 1, 0)

        # Set Dark pixels
        # Get B8_arr already previously descaled corresponding month tif (NIR)
        # NIR dark pixel reflectance threshold is set to 0.15 needs to be multiplied by scaling factor
        SR_BD_SCALE = 0.0001
        nir_drk_thresh = 0.15
        dark_pixels = (B8_arr < (nir_drk_thresh * SR_BD_SCALE)) * not_water
























        # Apply mask can be done after creating MLEONN products
        # # apply mask before scale factor to not change the acm_mask 1 and 999 values
        # # Set a threshold for cloud probability
        # # MSK_CLDPRB valid values from 0 to 100 (100 % probability of pixel being cloudy)
        # threshold = 20
        # ds_masked = ds.where(ds['MSK_CLDPRB_20m'].values <= threshold, np.nan)
        #
        #     # # create an xarray for the month containing all bands
        #     ds, dts, ys, xs = create_ds(input_dict, band_names)


    # get config file name for the site and read start and end date
    # config_fname = glob.glob(f'{output_dir}{tile_name}/*.yml')[0]
    # process_MLEONN(lai, config_fname, dts, ys, xs, saved_path, tile_name, 'lai')
    # process_MLEONN(fapar, config_fname, dts, ys, xs, saved_path, tile_name, 'fapar')
    # process_MLEONN(fc, config_fname, dts, ys, xs, saved_path, tile_name, 'fc')
    # process_MLEONN(cab, config_fname, dts, ys, xs, saved_path, tile_name, 'cab')


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <S2_path>")  # the user has to input one argument
    else:
        # location of the second item in the list
        # Sentinel 2 folder path where all bands folders are
        S2_path = sys.argv[1]
        main(S2_path)
