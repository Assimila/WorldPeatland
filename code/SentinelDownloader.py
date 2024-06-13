import numpy as np
import datetime as dt
import xarray as xr
import os
import ee
import osgeo
import subprocess
import shapely
import rioxarray    
import shapely
import shapely.geometry
import glob
import time
import sys
from save_xarray_to_gtiff import save_xarray
import pandas as pa
import copy

import warnings
warnings.filterwarnings('ignore')
    
try:
    import ogr
    import osr
    import gdal
except:
    from osgeo import ogr
    from osgeo import osr
    from osgeo import gdal

try:
    from ShapefileToolbox import ShapefileToolbox
except:
    try:
        os.path.append('../')
        from ShapefileToolbox import ShapefileToolbox
    except:
        raise ValueError('Could not find ShapefileToolbox in either:\n[%s, %s]'%(os.getcwd(),
                                                                                os.path.join(os.getcwd(),'..')))

class SentinelDownloader():
    
    """
    SentinelDownloader
    Downloading module for getting all combinations of relevant Sentinel data.
    Improvments over previous versions is ACM is included download is included
    in the code base, no layers discarded when writing to a datacube so all
    Sentinel 2 products can be saved in the same shape and no mis-alignment is
    possible and downloaded files returned as a list so they can be deleted
    after datacube ingestion. 
    
    Usage:
    # to download data, specifiy any shapefile of any shape and a time window.
    # the shapefiel will be buffered by 50m (can be changed by editing the 
    # attributes of the class). 
    sd = SentinelDownloader('/media/DataShare/Alex/grip/downloader_test_chip.shp',
                             dt.date(2022,1,1), dt.date(2022,12,31))
    # to download S1, S2 & ACM, use 'download_raw_all'. The new annual files
    # will get listed in the return of the method. Files are downloaded to the
    # only argument of the method.
    new_dwn_files = sd.download_raw_all('/data/grip/staging/')
    
    # these file then get ingested into the datacube, where the second
    # argment is the root of the datacube. 
    sd.write_raw_files_to_datacube(new_dwn_files,'/data/grip/datacube/')

    # only download s1
    new_dwn_files = sd.download_raw_s1('/data/grip/staging/')
    sd.write_raw_files_to_datacube(new_dwn_files,'/data/grip/datacube/')
    
    # only download s2 & ACM
    new_dwn_files = sd.download_raw_s2('/data/grip/staging/')
    sd.write_raw_files_to_datacube(new_dwn_files,'/data/grip/datacube/')
    
    # only download s2 (no acm)
    new_dwn_files = sd.download_raw_s2('/data/grip/staging/',get_acm=False)
    sd.write_raw_files_to_datacube(new_dwn_files,'/data/grip/datacube/')
    """
    
    def __init__(self, shapefile, start, end, ee_credentials = None, project = None):
        
        self.shapefile = shapefile
        self.start = start
        self.end = end
        self.verbose = True
        
        # number of meters to buffer around the field
        self.buffer_meters = 50
        self.override_existing_files = False
        self.write_util_files = True
        self.project = project
        
        self._ee_auth(ee_credentials)
        
        self.acm_params = {'CLD_PRB_THRESH': 45,
                           'NIR_DRK_THRESH': 0.25,
                           'CLD_PRJ_DIST': 2,
                           'BUFFER': 70}

    def _ee_auth(self, ee_credentials):
        
        if ee_credentials is None:
            try:
                if self.project is None:
                    ee.Initialize()
                else:
                    ee.Initialize(project=self.project)
            except Exception as e:
                ee.Authenticate()
                ee.Initialize()
        else:
            ee.Initialize(ee_credentials)
        
    def __prep_download_bounding_box(self):
        """
        Method to prepare the bounding box in WGS84 and make sure there
        is a directory to put the utility shapefiles
        """
        
        util_dir = os.path.join(os.path.dirname(self.shapefile),
                                          'util_shapefiles')
        if os.path.isdir(util_dir) == False:
            os.makedirs(util_dir)
            
        reproj = os.path.join(util_dir,
                              self.shapefile.split('/')[-1].replace('.shp','_reproj.shp'))
        buffered = reproj.replace('_reproj.shp','_bb.shp')
        
        shpt = ShapefileToolbox(self.shapefile)
        outline_utm = shpt.get_utm_perimeter()
        if self.write_util_files:
            shpt.write_output(outline_utm, reproj)
        
        bbox = shpt.get_wgs84_envelope(buffer_meters=self.buffer_meters)

        if self.write_util_files:
            shpt.write_output(bbox, buffered)
        self.epsg_code = shpt.epsg_code
        
        bb = {'north': max(bbox['lat']),
             'south': min(bbox['lat']),
             'west': min(bbox['lon']),
             'east': max(bbox['lon'])}
        
        return bb
    
    def __gen_download_key(self):
        """
        Method to create a unique handle for downloaded data so when writing 
        files to the datacube, only the new files fitting the specific 
        pattern are processed.
        """
        
        key = self.shapefile.split('/')[-1].replace('.geojson','')
        key = self.shapefile.split('/')[-1].replace('.shp','')
        self.key = key
        
        time_key = key + '_%s'%dt.datetime.now().strftime('%Y%m%d%H%M%S')
        
        return time_key

    def write_raw_files_to_datacube(self, new_file_list, datacube_root):
        """
        Method to ingest the freshly downloaded files to a datacube structure.
        """
        for i in new_file_list:
            
            print(i)

            print('+++++++++++++++++++++++++++++++++++++++++++++')


            if 'S1_GRD' in i:
                self.__make_s1_datacube_geotiffs(i,datacube_root,'20240611090406') #self.key)
            if 'S2_SR' in i:
                self.__make_s2_datacube_geotiffs(i,datacube_root,'20240611090406')#self.key)
            if 'ACM' in i:
                self.__make_cloudmask_datacube_geotiffs(i,datacube_root,'20240611090406') #self.key)
        
    def download_raw_s2(self, staging_area, get_acm = True, manual_key = None):
        
        # find the bounding box and download key
        bb = self.__prep_download_bounding_box()
        if manual_key is None:
            download_key = self.__gen_download_key()
        else:
            self.key = manual_key
            download_key = self.key + '_%s'%dt.datetime.now().strftime('%Y%m%d%H%M%S')
        s2_staging_dir = os.path.join(staging_area,'s2')
        
        # download the data 
        self._download_s2_data(bb, self.start, self.end, download_key, s2_staging_dir)
        
        # find the new files
        new_s2_files = sorted(glob.glob(os.path.join(s2_staging_dir,
                                                     'S2_SR_*%s*tif'%download_key))) 
        
        # download the ACM if needed
        if get_acm:
            acm_staging_dir = os.path.join(staging_area,'acm')
            self._download_acm_data(bb, self.start, self.end, 
                                    download_key, acm_staging_dir)
            new_acm_files = sorted(glob.glob(os.path.join(acm_staging_dir,
                                                          'ACM_*%s*tif'%download_key)))
            
            return np.concatenate([new_s2_files, new_acm_files])
        else:
            return new_s2_files
        
    def download_raw_acm(self, staging_area):
        
        # find the bounding box and download key
        bb = self.__prep_download_bounding_box()
        download_key = self.__gen_download_key()
        acm_staging_dir = os.path.join(staging_area,'acm')
        
        # download the files and retunr them for datacube writing
        self._download_acm_data(bb, self.start, self.end, download_key, acm_staging_dir)
        new_acm_files = sorted(glob.glob(os.path.join(acm_staging_dir,
                                                      'ACM_*%s*tif'%download_key)))
            
        return new_acm_files
        
            
    def download_raw_s1(self, staging_area, manual_key = None):
        
        # find the bounding box and download key
        bb = self.__prep_download_bounding_box()
        if manual_key is None:
            download_key = self.__gen_download_key()
        else:
            self.key = manual_key
            download_key = self.key + '_%s'%dt.datetime.now().strftime('%Y%m%d%H%M%S')  
            
        s1_staging_dir = os.path.join(staging_area,'s1')        
        
        # download the files and retunr them for datacube writing
        self._download_s1_data(bb,self.start,self.end,download_key,s1_staging_dir)
        new_s1_files = sorted(glob.glob(os.path.join(s1_staging_dir,'S1_GRD_*%s*tif'%download_key)))
        
        return new_s1_files
        
    def download_raw_all(self, staging_area, manual_key = None): # staging_area set by user 
       
        # find the bounding box and download key
        bb = self.__prep_download_bounding_box()
        if manual_key is None:
            download_key = self.__gen_download_key()
        else:
            self.key = manual_key
            download_key = self.key + '_%s'%dt.datetime.now().strftime('%Y%m%d%H%M%S')
        
        s1_staging_dir = os.path.join(staging_area,'s1')
        s2_staging_dir = os.path.join(staging_area,'s2')
        acm_staging_dir = os.path.join(staging_area,'acm')
        
        # download all three datasets
        self._download_s1_data(bb, self.start, self.end, download_key,
                               s1_staging_dir)
        self._download_s2_data(bb, self.start, self.end, download_key,
                               s2_staging_dir)
        self._download_acm_data(bb, self.start, self.end, download_key,
                                acm_staging_dir)
        
        # return all the new files
        new_s1_files = sorted(glob.glob(os.path.join(s1_staging_dir,
                                                     'S1_GRD_*%s*tif'%download_key)))
        new_s2_files = sorted(glob.glob(os.path.join(s2_staging_dir,
                                                     'S2_SR_*%s*tif'%download_key)))
        new_acm_files = sorted(glob.glob(os.path.join(acm_staging_dir,
                                                      'ACM_*%s*tif'%download_key)))
        
        return np.concatenate([new_s1_files, new_s2_files, new_acm_files])
        
        
    def __find_annual_download_chunks(self, download_start, download_end):
        
        """
        Function to find the strings to specifiy the annual date ranges
        """
        year_ranges = []
        
        # find the difference in years between the start and end
        year_dif = download_end.year - download_start.year
        # easy option is that the start and end are in the same year
        if year_dif == 0:
            year_ranges.append([download_start.strftime('%Y-%m-%d'),
                               download_end.strftime('%Y-%m-%d')])
        # stick two year ranges together
        elif year_dif == 1:
            year_ranges.append([download_start.strftime('%Y-%m-%d'),
                               '%s-12-31'%download_start.year])
            year_ranges.append(['%s-01-01'%download_end.year,
                                download_end.strftime('%Y-%m-%d')])
        # if more than two years, loop throguh each year 
        else:
            for i in range(year_dif+1):
                if i == 0:
                    year_ranges.append([download_start.strftime('%Y-%m-%d'),
                                        '%s-12-31'%download_start.year])
                elif i == year_dif:
                    year_ranges.append(['%s-01-01'%download_end.year,
                                        download_end.strftime('%Y-%m-%d')])
                else:
                    year_ranges.append(['%s-01-01'%str(download_start.year + i),
                                       '%s-12-31'%str(download_start.year + i)])
                    
        return year_ranges
        
        
    def _download_s1_data(self, in_boundary, download_start, download_end,
                         key, download_directory):
        
        # put the bounding box into a geojson which is what earth engine works with
        geoj = self.__gen_geojson(in_boundary['north'],
                           in_boundary['south'],
                           in_boundary['east'],
                           in_boundary['west'])

        # create an earth engine object that will deliniate where to download
        geometry = ee.Geometry.Polygon(geoj['features'][0]['geometry']['coordinates'])
        
        
        
        # specifiy for earth engine which product we are downloading
        product_name = 'S1_GRD'

        # Product using GEE nomenclature
        product = f'COPERNICUS/{product_name}'

        # download individual years as smaller requests go faster
        year_ranges = self.__find_annual_download_chunks(download_start, download_end)
        
        for start_date,end_date in year_ranges:
            
            year = start_date.split('-')[0]

            # Construct the GEE collection
            bands = ['VH','VV','angle']
            resolutions = [10,10,10]
            # Get GEE collection  # TODO apply filter edge polarization
            ee_collection = (ee.ImageCollection(product).
                             filterBounds(geometry).
                             filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).
                             filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).
                             filterDate(start_date, end_date).
                             select(bands))

            # repository to put the tasks so they can be checked
            tasks = []
               
            # name of the Assimila Bucket where to put the data
            bucket = 'worldpeatland'

            # loop through the bands and export them individually as we want 
            # seperate files for each band
            for resolution, band in zip(resolutions, bands):                
                unique_desc = f'{key}_{band}'
                tasks.append(ee.batch.Export.image.toCloudStorage(
                    image=ee_collection.select(band).toBands().toFloat(),
                    region=geometry,
                    description=unique_desc,
                    bucket=bucket,
                    crs='EPSG:%s'%self.epsg_code,
                    fileNamePrefix='%s_%s_%s_%s'%(product_name, key, year, band),
                    scale=resolution,
                    maxPixels=1e13))
                
            if len(tasks) == 0:
                continue
            print(tasks) 
            # start the tasks
            for task in tasks:
                task.start()
            
            # continue in this while loop until all the tasks are finished. Once
            # complete they will be sitting on the google bucket
            start = time.time()
            while len([i.status()['state'] for i in tasks if i.status()['state'] == 'COMPLETED']) < len(bands):
                
                if self.verbose == True:
                    # Project must be setup in advance e.g
                    # earthengine set_project worldpeatland
                    result = subprocess.run(['earthengine', 'task', 'list'], stdout=subprocess.PIPE)
                    output = str(result.stdout).replace('  Export.image  ','').split('\\n')[:3]
                    trimmed = [[i for i in j.split(' ') if len(i) > 0] for j in output]
                    prnt_lines = ['%s - %s'%(i[0],i[1]) for i in trimmed]

                    second_dif = str(int(time.time() - start))+'s'
                    second_dif_str_second = f"{second_dif:<5}"
                    second_dif_str = '(S1 %s %s) %s'%(key, year,second_dif_str_second)
                    prnt = '%s - %s // %s // %s'%(second_dif_str,prnt_lines[0][2:],prnt_lines[1],prnt_lines[2])

                    sys.stdout.write('%s\r' % (prnt,))
                    sys.stdout.flush()
                else:
                    time.sleep(2)
            if self.verbose == True:
                print ('\n')

            # generate the destiation of where to put the data
            dest = os.path.join(download_directory)
            if os.path.isdir(dest) == False:
                os.makedirs(dest)
            # use command line gsutil to move them 
            cmd = 'gsutil mv gs://%s/%s_%s_%s* %s/'%(bucket, product_name, key, year, dest)

            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # get the metadata from the ee_collection

            # sensing start times, sensing end times
            s, e = self.__get_s1_sensing_times(ee_collection)
            # ascending/descending flags
            asds = self.__get_s1_orbit_type(ee_collection)

            # put the metadata in a dictionary in the right format
            meta_dict = {'sensingStartTime': [i.strftime('%Y-%m-%d_%H:%M:%S') for i in s], 
                        'sensingEndTime': [i.strftime('%Y-%m-%d_%H:%M:%S') for i in e],
                         'ascending/descending': asds}

            # add the metadata to the files 
            files2edit = sorted(glob.glob(os.path.join(dest, '%s_%s_%s_*tif'%(product_name, key, year))))
            [self.__write_geotiff_metadata(i, meta_dict) for i in files2edit]
    
    
    def _download_s2_data(self,in_boundary, download_start, download_end,
                         key, download_directory):
    
        # put the bounding box into a geojson which is what earth engine works with
        geoj = self.__gen_geojson(in_boundary['north'],
                           in_boundary['south'],
                           in_boundary['east'],
                           in_boundary['west'])

        # create an earth engine object that will deliniate where to download
        geometry = ee.Geometry.Polygon(geoj['features'][0]['geometry']['coordinates'])

        # Get Sentinel-2 MSI data
        product_name = 'S2_SR_HARMONIZED'

        # Product using GEE nomenclature
        product = f'COPERNICUS/{product_name}'

        # download individual years as smaller requests go faster
        year_ranges = self.__find_annual_download_chunks(download_start, download_end)
        
        for start_date,end_date in year_ranges:
        
            year = start_date.split('-')[0]
            
            # Specify all the bands we band to download - lots of S2!
            bands = ['B1','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

            # some bands (generally the longer wavelength bands) have lower native spatial
            # resolution. To prevent the earth engine getting really slow with un-needed
            # preprocessing on its side with resampling, we download everythinhg at its
            # native reoslution and then we resample the data on our side.
            resolutions = [60,10, 10, 10, 20, 20, 20, 10, 20, 20, 20]

            # Get GEE collection
            ee_collection = (ee.ImageCollection(product).
                             filterBounds(geometry).
                             filterDate(start_date, end_date).
                             select(bands))
            
            # filter the data so we only have one tile of data to avoid duplicates
            try:
                tile_names = ee_collection.aggregate_array('MGRS_TILE').getInfo()
                ee_collection = ee_collection.filterMetadata('MGRS_TILE', 'equals', tile_names[0])
            except:
                print ('Could not filter MGRS tiles - processing without filtering data.')
                pass

            # get the times that the images were aquired
            aquis_times = {'aquisTime': [i.strftime('%Y-%m-%d_%H:%M:%S') 
                                         for i in self.__get_s2_aquis_dates(ee_collection)]}
            
            # repository to put the active tasks in to check them
            tasks = []

            # specify the Assimila Google bucket to download the data to 
            bucket = 'worldpeatland'
            skipped_bands = 0

            # loop throguh each band to download a single file per band
            for n,band in enumerate(bands):
                
                unique_desc = f'{key}_{band}'
                file_name_prefix = '%s_%s_%s_%s'%(product_name, key, year, band)
                task_download = ee.batch.Export.image.toCloudStorage(
                    image=ee_collection.select(band).toBands(),
                    region=geometry,
                    description=unique_desc,
                    bucket=bucket,     
                    crs='EPSG:%s'%self.epsg_code,
                    fileNamePrefix=file_name_prefix,
                    scale=resolutions[n],
                    maxPixels=1e13)

                tasks.append(task_download)
            
            # start the tasks
            [i.start() for i in tasks]
            
            # wait on them to complete - the i.status()['state] will be 'COMPLETED' once
            # the data is in the google bucket
            start = time.time()
            while len([i.status()['state'] for i in tasks if i.status()['state'] == 'COMPLETED']) < len(bands):
                
                if self.verbose == True:
                    result = subprocess.run(['earthengine', 'task', 'list'], stdout=subprocess.PIPE)
                    output = str(result.stdout).replace('  Export.image  ','').split('\\n')[:10]
                    trimmed = [[i for i in j.split(' ') if len(i) > 0] for j in output]
                    prnt_lines = ['%s - %s'%(i[0],i[1]) for i in trimmed]

                    second_dif = str(int(time.time() - start))+'s'
                    second_dif_str_second = f"{second_dif:<5}"
                    second_dif_str = '(S2 %s %s) %s'%(key,year,second_dif_str_second)
                    prnt = '%s - %s // %s // %s // %s // %s // %s // %s // %s // %s // %s'%(second_dif_str,prnt_lines[0][2:],
                                                                                            prnt_lines[1],prnt_lines[2],
                                                                                            prnt_lines[3],prnt_lines[4],
                                                                                            prnt_lines[5],prnt_lines[6],
                                                                                            prnt_lines[7],prnt_lines[8],
                                                                                            prnt_lines[9])
                    sys.stdout.write('%s\r' % (prnt,))
                    sys.stdout.flush()
                else:
                    time.sleep(2)
            if self.verbose == True:
                print ('\n')
                
            # create the download directory if it does not exist
            dest = os.path.join(download_directory)
            if os.path.isdir(dest) == False:
                os.makedirs(dest)
            
            # use command line gsutils to move them out the bucket
            cmd = 'gsutil mv gs://%s/%s_%s_%s* %s/'%(bucket,product_name, key, year, dest)
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            
            # pull out the band image collection for the earth image collection
            # use any of the 10m bands to make files containing the angle data 
            angle_mimic = 'B2'
            ee_bcol = ee_collection.select(angle_mimic)

            # get the reflectance band data file so we have the shape, projection ect
            example_file = sorted(glob.glob(os.path.join(dest,'S2_*%s_%s_%s.tif'%(key,year,
                                                                                  angle_mimic))))[0]

            # create a lookup to convert from our nomenclature to earth engine nomenclature
            angle_keys = {'sza': 'MEAN_SOLAR_ZENITH_ANGLE',
                  'saa': 'MEAN_SOLAR_AZIMUTH_ANGLE',
                  'vza': 'MEAN_INCIDENCE_ZENITH_ANGLE_%s'%angle_mimic,
                  'vaa': 'MEAN_INCIDENCE_AZIMUTH_ANGLE_%s'%angle_mimic}

            # pull out the angles, which are image attributes, from the earth engine object
            ad = {key: ee_bcol.aggregate_array(angle_keys[key]).getInfo() for key in list(angle_keys.keys())}

            # get all the relevant info for generating a new file
            example_ds = gdal.Open(example_file)
            geot = example_ds.GetGeoTransform()
            proj = example_ds.GetProjection()
            cols = example_ds.RasterXSize
            rows = example_ds.RasterYSize
            lyrs = example_ds.RasterCount
            # these extrta option help compress the data and format them correctly for
            # so they can be put into the datacube.
            driver_options = ['COMPRESS=DEFLATE',
                              'BIGTIFF=YES',
                              'PREDICTOR=1',
                              'TILED=YES',
                              'COPY_SRC_OVERVIEWS=YES']

            # create the files for each angle type
            for j in angle_keys:

                # create a save name with the angle type
                save_name = example_file.replace('_%s.tif'%angle_mimic, '_%s.tif'%j)
                
                # create a blank geotiff
                driver = gdal.GetDriverByName('GTiff')
                # dataype can be int16 to save space and we dont need more precision
                outRaster = driver.Create(save_name, cols, rows, lyrs, gdal.GDT_Int16, driver_options)
                # set the projection info
                outRaster.SetGeoTransform(geot)
                outRaster.SetProjection(proj)

                # loop throguh each of the layers and fill them with the angle data
                # that correspons with each of the layers
                for k in range(1,lyrs+1):
                    # ad[j][k-1] is just one number e.g. 27
                    data_lyr = np.zeros([rows, cols])+ad[j][k-1]
                    # write it
                    outband = outRaster.GetRasterBand(k)
                    outband.WriteArray(data_lyr)
                # write the data to disk
                outband.FlushCache()
                outband = outRaster = None
                # the file is now written
            
            # get a list of all the newly downloaded files
            files2edit = sorted(glob.glob(os.path.join(dest, '%s_%s_%s_*tif'%(product_name, key, year))))
            # add the metadata to the files
            [self.__write_geotiff_metadata(i, aquis_times) for i in files2edit]           
            
    def _download_acm_data(self,in_boundary, download_start, download_end,
                         key, download_directory):
        
        # put the bounding box into a geojson which is what earth engine works with
        geoj = self.__gen_geojson(in_boundary['north'],
                           in_boundary['south'],
                           in_boundary['east'],
                           in_boundary['west'])

        # create an earth engine object that will deliniate where to download
        geometry = ee.Geometry.Polygon(geoj['features'][0]['geometry']['coordinates'])
        # The size of the buffer required relates to the length you're projecting the cloud.  
        # If the projection distance is 2km, say, then you need a buffer of at least 2 km to
        # encompass all of the clouds that could project onto our tile.I added a little margin
        # on top, so if the projection distance was 2km, we could buffer by 2.1km. 
        # (the 10% was on the projection distance, not the field size)
        
        # buffer the geometry by the KM + 10%
        geometry_b = geometry.buffer((self.acm_params['CLD_PRJ_DIST']*1000)*1.1)
        
        product_name = 'ACM'

        # download individual years as smaller requests go faster
        year_ranges = self.__find_annual_download_chunks(download_start, download_end)
        
        for start_date,end_date in year_ranges:
           
            year = start_date.split('-')[0]
            
            # get the S2 bands needed for the cloud masking
            bands = ['B8','SCL']
            
            # create a feature collection that is buffered by the cloud projection 
            s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(geometry_b)
                        .filterDate(start_date, end_date)
                        .select(bands))
            
            # remove duplicates of data in different tiles
            tile_names = s2_sr_col.aggregate_array('MGRS_TILE').getInfo()
            s2_sr_col = s2_sr_col.filterMetadata('MGRS_TILE', 'equals', tile_names[0])
            
            # Import and filter s2cloudless with the buffered outline
            s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                .filterBounds(geometry_b)
                .filterDate(start_date, end_date))
            
            # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
            filter_dict = {'primary': s2_sr_col,'secondary': s2_cloudless_col,
                            'condition': ee.Filter.equals(**{'leftField': 'system:index','rightField': 'system:index'})}
            
            # merge everything together
            basic_collection = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**filter_dict))
            
            # do the cloud masking by mapping the internal function to the full datset
            ee_collection = basic_collection.map(self.__add_cld_shdw_mask)   
            
             # get the times that the images were aquired
            aquis_times = {'aquisTime': [i.strftime('%Y-%m-%d_%H:%M:%S') 
                                         for i in self.__get_s2_aquis_dates(basic_collection)]}
            
            # repository to put the active tasks in to check them
            tasks = []

            # specify the Assimila Google bucket to download the data to 
            bucket = 'worldpeatland'
            skipped_bands = 0
            
            bands = ['clouds','cloud_transform','dark_pixels','probability','shadows','cloudmask']
            resolutions = [10,10,10,10,10,10]
            
            # https://gis.stackexchange.com/questions/461913/reproject-and-export-data-is-not-reprojected-in-google-earth-engine
            
            # loop throguh each band to download a single file per band
            for n,band in enumerate(bands):

                unique_desc = f'{key}_{band}'
                file_name_prefix = '%s_%s_%s_%s'%(product_name, key, year, band)
                task_download = ee.batch.Export.image.toCloudStorage(
                    image=ee_collection.select(band).toBands(),
                    region=geometry,
                    description=unique_desc,
                    bucket=bucket,     
                    crs='EPSG:%s'%self.epsg_code,
                    fileNamePrefix=file_name_prefix,
                    scale=resolutions[n],
                    maxPixels=1e13)

                tasks.append(task_download)
            
            # start the tasks
            [i.start() for i in tasks]
            
            # wait on them to complete - the i.status()['state] will be 'COMPLETED' once
            # the data is in the google bucket
            start = time.time()
            while len([i.status()['state'] for i in tasks if i.status()['state'] == 'COMPLETED']) < len(tasks):
                
                result = subprocess.run(['earthengine', 'task', 'list'], stdout=subprocess.PIPE)
                output = str(result.stdout).replace('  Export.image  ','').split('\\n')[:15]
                trimmed = [[i for i in j.split(' ') if len(i) > 0] for j in output]
                prnt_lines = ['%s - %s'%(i[0],i[1]) for i in trimmed[:-1]]
                second_dif = str(int(time.time() - start))+'s'
                second_dif_str_second = f"{second_dif:<5}"
                second_dif_str = '(ACM %s %s) %s'%(key,year,second_dif_str_second)
                prnt = '%s - %s // %s // %s // %s // %s // %s'%(second_dif_str,prnt_lines[0][2:],prnt_lines[1],prnt_lines[2],
                                                                prnt_lines[3],prnt_lines[4],prnt_lines[5])

                sys.stdout.write('%s\r' % (prnt,))
                sys.stdout.flush()
                
            print ('\n')

            # create the download directory if it does not exist
            dest = os.path.join(download_directory)
            if os.path.isdir(dest) == False:
                os.makedirs(dest)
            
            # use command line gsutils to move them out the bucket
            cmd = 'gsutil mv gs://%s/%s_%s_%s* %s/'%(bucket,product_name, key, year, dest)
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # get the times that the images were aquired
            aquis_times = {'aquisTime': [i.strftime('%Y-%m-%d_%H:%M:%S') 
                                         for i in self.__get_s2_aquis_dates(ee_collection)]}
            
            # get a list of all the newly downloaded files
            files2edit = sorted(glob.glob(os.path.join(dest, '%s_%s_%s_*tif'%(product_name, key, year))))
            # add the metadata to the files
            [self.__write_geotiff_metadata(i, aquis_times) for i in files2edit]
            
    def __gen_geojson(self,north,south,east,west):
        
        """
        Function to generate a geojson from the bounding coordinates
        of a region to download. 
        """
        geojson = """{"type": "FeatureCollection",
              "features": [{
                  "type": "Feature",
                  "properties": {},
                  "geometry": {
                    "type": "Polygon",
                    "coordinates": [  [  [%s,%s], [ %s, %s],
                                        [%s,%s], [%s, %s],
                                        [%s,%s] ] ]
                  }}]}"""%(west, south, east, south, 
                           east, north, west, north, 
                           west, south)
        return eval(geojson)
    
    def __get_s1_orbit_type(self,in_ee):

        """
        Function to get flags of if the satellite was ascending or descending at the time
        of aquisition. Returns a list of strings.
        INPUTS:
            - in_ee (object) - The google earth engine obecject for which you are downloading
                data for.
        """

        asds = in_ee.aggregate_array('orbitProperties_pass').getInfo()

        return asds
    
    def __get_s1_sensing_times(self,in_ee):

        """
        Function to get the times at which a Sentinel 1 image was aquired. 
        This works by getting the node orbit properties, which are in seconds past an epoch,
        and then adds a number of seconds onto this to define when the sensing started and 
        stopped. Returns two lists of datetimes.
        INPUTS:
            - in_ee (object) - The google earth engine obecject for which you are downloading
                data for.
        """
        # aggregate_array calls a list in which all the metadata properties for 
        # each layer within the collection are returned. 

        # get all the image metadata properties in one call and convert them to datetimes
        node_pass = [dt.datetime.fromtimestamp(i/1000) 
                     for i in in_ee.aggregate_array('orbitProperties_ascendingNodeTime').getInfo()]

        # add the respective lag times to the nodepasses
        sensing_start = [node_pass[n] + dt.timedelta(seconds=i/1000) 
                         for n,i in enumerate(in_ee.aggregate_array('startTimeANX').getInfo())]

        sensing_end = [node_pass[n] + dt.timedelta(seconds=i/1000) 
                         for n,i in enumerate(in_ee.aggregate_array('stopTimeANX').getInfo())]

        return sensing_start,sensing_end
    
    def __get_s2_aquis_dates(self,in_ee):
        """
        Function to get the aqusition datetimes of the imagery from an earth engine 
        object. Returns a list of dt.datetimes of the aquisition datetimes.
        INPUTS:
             - in_ee (object) - a google earth engine object to be get data from.
        """
        # specifiy the image attributes to get
        id_strip= 'DATASTRIP_ID'
        # get all the id_strips for each layer at once
        ids = in_ee.aggregate_array(id_strip).getInfo()
        # loop throguh and turn them to datetimes
        datetimes = [dt.datetime.strptime(i.replace('__','_').split('_')[6], '%Y%m%dT%H%M%S') for i in ids]
        return datetimes
    
    def __write_geotiff_metadata(self, in_file, meta_dict):
        """
        Function to add per-band metadata to a 3D geotiff.
        INPUTS:
            - in_file (string) - absolute path to the geotiff.
            - meta_dict (dictionary) - a dictionary where the keys are
                the names of metadata fields and each key indexes a list
                with all the metadata items. All lists must be the same size
                and must be equal in length to the number of layers in in_file.
        """

        # open the dataset
        opn = gdal.Open(in_file)

        # get the names of the metadata fields
        kys = list(meta_dict.keys())

        # loop through the layers - gdal starts counting layers from 1
        for i in range(1,opn.RasterCount+1):
            # get the layer
            rst = opn.GetRasterBand(i)
            # generate the per-band metadata dictionary by looping
            # through the keys.i-1 indexes the metadata for that layer
            meta2put = {key: meta_dict[key][i-1] for key in kys}

            # set the metadata and write it to disk
            rst.SetMetadata(meta2put)        
            rst.FlushCache()
            rst = None

        # close the dataset and make sure everything is written properly
        opn.FlushCache()
        opn = None
        
    def __make_s1_datacube_geotiffs(self, in_fname, output_dir, tile_name):

        opn = gdal.Open(in_fname)
        aquis = np.array([opn.GetRasterBand(i).GetMetadata()['sensingStartTime'] for i in range(1,opn.RasterCount+1)])
        orbit = np.array([opn.GetRasterBand(i).GetMetadata()['ascending/descending'] for i in range(1,opn.RasterCount+1)])
        sort_ind = np.argsort(aquis)

        md = pa.DataFrame({'timeFormat': aquis})
        md.timeFormat = pa.to_datetime(md['timeFormat'], format='%Y-%m-%d_%H:%M:%S').to_numpy()
        splt = in_fname.split('/')[-1].split('_')    
        product = splt[0]+'_'+splt[1]
        year = splt[-2]
        
        if xr.__version__ >= '0.20.0':
            # newer versions of xarray and rioxarray
            ds = rioxarray.open_rasterio(in_fname)
            ds.attrs['crs'] = ds.spatial_ref.attrs['spatial_ref']
        else:
            # older xarray handling
            ds = xr.open_rasterio(in_fname)

        # Rename dimensions
        ds = ds.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
        ds = ds.assign_coords(time=list(md.timeFormat))
        ds = ds.astype(np.float32)

        # sort the data so it is in chronological order
        ds = ds.isel({'time': sort_ind})

        orbit = orbit[sort_ind]

        ds_a = ds.isel({'time': np.where(orbit == 'ASCENDING')[0]})
        ds_d = ds.isel({'time': np.where(orbit == 'DESCENDING')[0]})

        for n,ds_orbit in enumerate([ds_a,ds_d]):

            subproduct = splt[-1].replace('.tif','') + ['_ASCENDING', '_DESCENDING'][n]

            # Change fill value. Original is 0. Note, 0 is also a valid value however
            # the S1 data also uses it as a fill value
            fill_value = 999
            ds_orbit = ds_orbit.where(ds_orbit != 0, fill_value)

            attrs = copy.deepcopy(ds_orbit.attrs)

            # loop throguh the individual months
            months = np.array([i.astype('datetime64[M]') for i in ds_orbit['time'].values])

            for month in np.unique(months):

                monthly_ind = np.where(months == month)[0]
                fname_month = (str(month).split('-')[1])
                fname_year = (str(month).split('-')[0])

                fname_ext = f'{product}_{subproduct}_{tile_name}_{fname_year}-{fname_month}.tif'
                dataDir = os.path.join(output_dir,product,subproduct,tile_name)
                fname = os.path.join(dataDir, fname_ext)
                if self.override_existing_files == False:
                    if os.path.isfile(fname) == True:
                        continue
   
                ds_orbit_monthly = ds_orbit.isel({'time': monthly_ind})
                # ds_orbit_monthly perform checker of corruputed data checker is a function 
                # input dataarray check for corrupted values....
                # between min and max otherwise DataArray.drop

                if os.path.isdir(dataDir) == False:
                    os.makedirs(dataDir)

                ds_orbit_monthly = ds_orbit_monthly.to_dataset(name='S1_GRD')

                # the above command does not retain the atributes of the dataset. So you
                # have to reset the attributes to the datarray

                ds_orbit_monthly.attrs = attrs
                #  Create metadata dictionary
                ds_md = {'add_offset': 0, 
                         'scale_factor': 1,
                         'fill_value': fill_value,
                         'version': 'S1TBX',
                         'product': product}

                save_xarray(fname, xarray=ds_orbit_monthly, data_var='S1_GRD', metadata=ds_md)
                
    def __make_s2_datacube_geotiffs(self, in_fname, output_directory, tile_name):
        
        opn = gdal.Open(in_fname)
        aquis = np.array([opn.GetRasterBand(i).GetMetadata()['aquisTime'] for i in range(1,opn.RasterCount+1)])
        sort_ind = np.argsort(aquis)

        md = pa.DataFrame({'timeFormat': aquis})
        md.timeFormat = pa.to_datetime(md['timeFormat'], format='%Y-%m-%d_%H:%M:%S').to_numpy()
        splt = in_fname.split('/')[-1].split('_')   
        product = splt[0]+'_'+splt[1]
        subproduct = splt[-1].replace('.tif','').replace('.','_')
        year = splt[-2]
        
        dataDir = os.path.join(output_directory,product,subproduct,tile_name)
        
        if xr.__version__ >= '0.20.0':
            # newer versions of xarray and rioxarray
            ds = rioxarray.open_rasterio(in_fname)
            ds.attrs['crs'] = ds.spatial_ref.attrs['spatial_ref']
        else:
            # older xarray handling
            ds = xr.open_rasterio(in_fname)

        # Rename dimensions
        ds = ds.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
        ds = ds.assign_coords(time=list(md.timeFormat))
        ds = ds.astype(np.uint16)

        # sort the data so it is in chronological order
        ds = ds.isel({'time': sort_ind})

        # Change fill value. Original is 0. Note, 0 is also a valid value however
        # the S1 data also uses it as a fill value
        fill_value = 999
        ds = ds.where(ds != 0, fill_value)

        # loop throguh the individual months
        months = np.array([i.astype('datetime64[M]') for i in ds['time'].values])

        for month in np.unique(months):

            monthly_ind = np.where(months == month)[0]
            fname_month = (str(month).split('-')[1])
            fname_year = (str(month).split('-')[0])

            fname_ext = f'{product}_{subproduct}_{tile_name}_{fname_year}-{fname_month}.tif'
            dataDir = os.path.join(output_directory,product,subproduct,tile_name)
            fname = os.path.join(dataDir, fname_ext)
            if self.override_existing_files == False:
                if os.path.isfile(fname) == True:
                    continue

            ds_monthly = ds.isel({'time': monthly_ind})

            attrs = copy.deepcopy(ds_monthly.attrs)

            if os.path.isdir(dataDir) == False:
                os.makedirs(dataDir)           

            ds_monthly = ds_monthly.to_dataset(name='S2_SR')

            # the above command does not retain the atributes of the dataset. So you
            # have to reset the attributes to the datarray

            ds_monthly.attrs = attrs
            #  Create metadata dictionary
            ds_md = {'add_offset': 0, 
                     'scale_factor': 1/10000,
                     'fill_value': fill_value,
                     'version': 'Sentinel-2_L2_Sen2Cor', 
                     'product': product}

            save_xarray(fname, xarray=ds_monthly, data_var='S2_SR',metadata=ds_md)
            
    def __make_cloudmask_datacube_geotiffs(self, in_fname, output_directory, tile_name):
    
        subproduct = in_fname.split('_')[-1].replace('.tif','')

        opn = gdal.Open(in_fname)
        aquis = np.array([opn.GetRasterBand(i).GetMetadata()['aquisTime'] for i in range(1,opn.RasterCount+1)])
        sort_ind = np.argsort(aquis)

        md = pa.DataFrame({'timeFormat': aquis})
        md.timeFormat = pa.to_datetime(md['timeFormat'], format='%Y-%m-%d_%H:%M:%S').to_numpy()
        splt = in_fname.split('/')[-1].split('_')   
        product = 'ACM'

        dataDir = os.path.join(output_directory,product,subproduct,tile_name)

        if xr.__version__ >= '0.20.0':
            # newer versions of xarray and rioxarray
            ds = rioxarray.open_rasterio(in_fname)
            ds.attrs['crs'] = ds.spatial_ref.attrs['spatial_ref']
        else:
            # older xarray handling
            ds = xr.open_rasterio(in_fname)

        # Rename dimensions
        ds = ds.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
        ds = ds.assign_coords(time=list(md.timeFormat))

        # sort the data so it is in chronological order
        ds = ds.isel({'time': sort_ind})

        # Change fill value. Original is 0. Note, 0 is also a valid value however
        # the S1 data also uses it as a fill value
        fill_value = 999
        ds = ds.where(ds != 0, fill_value)

        # loop throguh the individual months and year
        years = np.array([i.astype('datetime64[Y]') for i in ds['time'].values])
        months = np.array([i.astype('datetime64[M]') for i in ds['time'].values])

        for year in np.unique(years):
            for month in np.unique(months):

                monthly_ind = np.where((months == month) & (years == year))[0]
                if len(monthly_ind) == 0:
                    continue
                fname_month = (str(month).split('-')[1])
                fname_year = (str(month).split('-')[0])

                fname_ext = f'{product}_{subproduct}_{tile_name}_{fname_year}-{fname_month}.tif'
                dataDir = os.path.join(output_directory,product,subproduct,tile_name)
                fname = os.path.join(dataDir, fname_ext)
                
                ds_monthly = ds.isel({'time': monthly_ind})

                attrs = copy.deepcopy(ds_monthly.attrs)

                if os.path.isdir(dataDir) == False:
                    os.makedirs(dataDir)

                ds_monthly = ds_monthly.to_dataset(name='ACM')

                # the above command does not retain the atributes of the dataset. So you
                # have to reset the attributes to the datarray

                ds_monthly.attrs = attrs
                #  Create metadata dictionary
                ds_md = {'add_offset': 0, 
                         'scale_factor': 1,
                         'fill_value': fill_value,
                         'product': product,
                         'tutorial': 'https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless',
                         'CLD_PRB_THRESH': self.acm_params['CLD_PRB_THRESH'],
                         'NIR_DRK_THRESH': self.acm_params['NIR_DRK_THRESH'],
                         'CLD_PRJ_DIST': self.acm_params['CLD_PRJ_DIST'],
                         'BUFFER': self.acm_params['BUFFER']}
                

                save_xarray(fname, xarray=ds_monthly, data_var='ACM',metadata=ds_md)
                
    def __add_cloud_bands(self,img):
        
        #DEFAULT CLD_PRB_THRESH = 50
        CLD_PRB_THRESH = self.acm_params['CLD_PRB_THRESH']
        
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def __add_shadow_bands(self, img):
        
        #DEFAULT NIR_DRK_THRESH = 0,15
        NIR_DRK_THRESH = self.acm_params['NIR_DRK_THRESH']

        #DEFAULT CLD_PRJ_DIST= 1
        CLD_PRJ_DIST = self.acm_params['CLD_PRJ_DIST']
        
        # Identify water pixels from the SCL band.
        # scl is the landcover classification and 6 is water
        not_water = img.select('SCL').neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        # this returns a 1 if the B8 pixels are over the threshold - so likely cloud shadow
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        # this is just a single number
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
            .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
            .select('distance')
            .mask()
            .rename('cloud_transform'))

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))
            
    def __add_cld_shdw_mask(self, img):
        
        #DEFAULT BUFFER= 50 
        BUFFER = self.acm_params['BUFFER']
        
        # Add cloud component bands.
        img_cloud = self.__add_cloud_bands(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = self.__add_shadow_bands(img_cloud)

        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20)
            .reproject(**{'crs': img.select([0]).projection(), 'scale': 10})
            .rename('cloudmask'))

        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)
