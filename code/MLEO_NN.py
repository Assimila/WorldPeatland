import numpy as np
import datetime as dt
import pandas as pa
import copy
try:
    import gdal
except:
    from osgeo import gdal
import glob
try:
    import osr
except:
    from osgeo import osr
try:
    import ogr
except:
    from osgeo import ogr
import scipy
import scipy.ndimage
import xarray as xr
import os
import rioxarray

from save_xarray_to_gtiff import save_xarray

class MLEO_operator():
    
    """
    Class to orchestrate the compilation of the S2 data for use
    with the MLEO neural networks. The class uses the datacube structure,
    where the shapefile and datacube root is specified. The S2 and ACM data is
    opened and prepped. This can then be passed to the public methods that apply
    the SNAP neural networks to derive biophysical parameters. 
    
    EXAMPLE:
    import matplotlib.pyplot as plt
    
    tile = 'NUEF004'

    ###########################
    # V1
    ###########################
    # no masking and no shapefile applied
    # if the state_shapefile argument is None, then the whole
    # dataset will be opened up without clipping.

    obj_v1 = MLEO_operator('/data/NUE_Profits/datacube/', tile, 
                        None, # here is the state shapefile argument
                        apply_any_mask=False)

    # just return the 3d dataset
    lai_v1 = obj_v1.calc_lai()
    # write it to the datacube with a specific handle
    obj_v1.calc_write_microcubes('lai','NUEF004', 
                                   subproduct_alternate_name='lai_raw')

    # using the following will just save the subproduct name as 'lai'
    # obj_v1.calc_write_microcubes('lai','NUEF004')

    ###########################
    # V2
    ###########################

    # a shapefile is used but no masking is applied
    obj_v2 = MLEO_operator('/data/NUE_Profits/datacube/', tile, 
                        '/data/NUE_Profits/shapefiles/%s.shp'%tile,
                        apply_any_mask=False)

    lai_v2 = obj_v2.calc_lai()
    # use this to write this version to the datacube
    # note that it will overwite all data in the datacube subproduct
    # directory with that tile name, so use a bespoke subproduct name
    obj_v2.calc_write_microcubes('lai','NUEF004',
                                subproduct_alternate_name='lai_clipped')

    ###########################
    # V3
    ###########################

    # a shapefile is used AND ACM is applied
    obj_v3 = MLEO_operator('/data/NUE_Profits/datacube/', tile, 
                        '/data/NUE_Profits/shapefiles/%s.shp'%tile,
                        ACM=True)

    lai_v3 = obj_v3.calc_lai()
    # use this to write this version to the datacube
    obj_v3.calc_write_microcubes('lai','NUEF004',
                                subproduct_alternate_name='lai_clipped_acm_applied')


    # display the differences
    num = 10
    fig,axs = plt.subplots(1,3)
    axs[0].imshow(lai_v1[num])
    axs[1].imshow(lai_v2[num])
    axs[2].imshow(lai_v3[num])
    [i.grid(c='w') for i in axs]
    plt.tight_layout()
    
    """
    
    def __init__(self, tile_data_path,tile_name,
                 state_shapefile, apply_any_mask = True,
                 ACM=True, temporalCM=False, mask_bright=False):

        # Here is where you set the bands you are interested in
        self.band_map = ['02', '03', '04', '05', '06', '07',
                          '08', '8A', '11','12']
            
        self.angle_keys = ['vza','sza', 'vaa','saa']
        
        # set variables
        self.tile_data_path = tile_data_path
        self.tile_name = tile_name
        self.state_shapefile = state_shapefile    
        self.apply_any_mask = apply_any_mask
        
        # do certain prep steps
        self._get_dates()        
        self._gather_data()        
        self._build_lookup()
        if apply_any_mask:
            self._remove_fully_masked(ACM, temporalCM, mask_bright=False)
        
    def _acm_mask_data(self):
    
        cm_files = sorted(glob.glob(os.path.dirname(self.data_file_paths['02'][0].
                                    replace('S2_SR','ACM').
                                    replace('B2','cloudmask')) + '/ACM*tif'))

        cm_dates = {}    
        for i in cm_files:

            cm_opn = self._reproject_image(i,nodataval=2)

            cm_date_strs = [cm_opn.GetRasterBand(j+1).GetMetadata()['time']
                            for j in range(cm_opn.RasterCount)]        
            cm_arr = cm_opn.ReadAsArray().astype(np.float32)
            cm_arr[cm_arr == 999] = 0
            cm_arr[cm_arr == 2] = np.nan
            
            if np.ndim(cm_arr) == 2:
                cm_arr = np.array([cm_arr])
                
            boundary = np.where(self.statemask == False)
            cm_arr[:, boundary[0], boundary[1]] = np.nan 

            for n,j in enumerate(cm_date_strs):
                aquis_date = dt.datetime.strptime(j.split('.')[0],'%Y-%m-%dT%H:%M:%S')
                cm_dates[aquis_date] = cm_arr[n] 
        
        self.cm = cm_dates
                
        for num, i in enumerate(self.dates):

            if i in list(cm_dates.keys()):

                to_mask = np.where(cm_dates[i] == 1)
                    
                if len(to_mask[0]) == 0:
                    continue
                if len(to_mask) == 1:
                    continue
                    
                band_to_mask = self.lookup[i]            
                for j in self.band_data:
                    self.band_data[j][:,band_to_mask,to_mask[0],to_mask[1]] = 0
        
    def _remove_fully_masked(self,  ACM=True, temporalCM=False, mask_bright=False):
        ## TODO: If temporal mask is false, then critical cloud percentage cover is not applied
        ##  
        if ACM and temporalCM:
            raise(ValueError, "Cannot apply the ACM and temporal mask together as currently implemented")
        
        
        if ACM:
            self._acm_mask_data()
        
        # then elimiate data         
        # find the mean profile of the first band
        mu = np.nanmean(self.band_data['data'][0],axis=(1,2))
        
        # no data steps have a mu of 0, so find where this
        # is not true
        has_data = np.where((mu > 0.) == True)[0]
        
        # loop throguh data and angles
        for i in np.concatenate([self.angle_keys,['data']]):
            
            # and cut to the data steps
            
            self.band_data[i] = np.array(self.band_data[i])[:,has_data]
        
        # and update the lookup
        good_dates = np.array(sorted(list(self.lookup.keys())))[has_data]
        
        self.lookup = dict(zip(good_dates,np.arange(len(good_dates))))
        
        # as well as the list of dates available
        self.dates = good_dates
        
        # then, use the method for detected clouds is:
        # Candra, D. S., Phinn, S., & Scarth, P. (2020). 
        # Cloud and cloud shadow masking for Sentinel-2 using multitemporal images 
        # in global area. International Journal of Remote Sensing, 41(8), 2877-2904.
        
        # not masking based off cloud shadow

        # find the S2 blue, red and green bands
        bands_for_masking = ['02','03','04']
        band_indexs = [n for n,i in enumerate(self.band_map) if i in bands_for_masking]
        if len(band_indexs) != 3:
            raise ValueError('Red, Green or Blue bands not present in band map')

        # put them into a 4d array
        targ = np.copy(np.array([self.band_data['data'][band_indexs[0]],
                         self.band_data['data'][band_indexs[1]],
                         self.band_data['data'][band_indexs[2]]]))

        targ[targ == 0] = np.nan
        # make the composite, so each pixel has outliers removed by the minimum filter and
        # averaged over the entire timeseries
        ref = np.nanmean(np.array([scipy.ndimage.minimum_filter1d(i,5,axis=0)
                            for i in np.copy(targ)]),
                    axis = 1)

        # find the difference between each pixel at each time and reference
        dif = np.array([i - ref[n] for n,i in enumerate(targ)])

        mask = np.zeros_like(targ[0])
        if temporalCM:
            # deliniate clouds
            mask[(dif[0] > 1000) & 
                   (dif[1] > 800) &
                   (dif[2] > 800)&
                   (targ[0] > 2000)] = 1
        elif mask_bright:
            # deliniate very bright pixels
            mask[(targ[0] > 2000)] = 1

        boundary = np.where(self.statemask == False)
        mask[:, boundary[0], boundary[1]] = np.nan    

        # find the timesteps with less than the critical amount of clouds
        percent_cloudy = np.nanmean(mask, axis=(1,2))

        #if maskcloud:
        
        crit = 1.0 #0.3
        has_data = np.where((percent_cloudy < crit) == True)[0]
        mask_trimmed = mask[has_data]
        pixel_mask = np.where(mask_trimmed == 1)
                
        # loop throguh data and angles
            
        for i in np.concatenate([self.angle_keys,['data']]):

            # and cut to the data steps
            self.band_data[i] = np.array(self.band_data[i])[:,has_data]
            self.band_data[i][:,pixel_mask[0],pixel_mask[1],pixel_mask[2]] = 0

        # and update the lookup
        good_dates = np.array(sorted(list(self.lookup.keys())))[has_data]

        self.lookup = dict(zip(good_dates,np.arange(len(good_dates))))

        # as well as the list of dates available
        self.dates = good_dates
            

    def _get_dates(self):
        
        # method to get the dates of each of the layers from the metadata
        
        product_dir = os.path.join(self.tile_data_path,'S2_SR')
        self.data_file_paths = {}
        
        # setup placeholder attributes
        # these will be found later
        self.base_file = None
        self.base_file_res = 1000000
        
        # loop through each of the bands to find the file paths
        for n,i in enumerate(self.band_map):
            
            subproduct = 'B'+i
            subproduct = subproduct.replace('B0','B')
            
            # get the directory for the band from the datacube
            data_dir = os.path.join(product_dir,subproduct,self.tile_name)
            
            # get the files in the directory
            monthly_files = sorted(glob.glob(os.path.join(data_dir,'S2_SR*')))
            monthly_files = [j for j in monthly_files if 'aux.xml' not in j]
            
            # on the first iteration, get the dates from the files.
            if n == 0:
                aquis_dates = []
                
                # loop through the files                
                for j in monthly_files:
                    
                    # open them and get the metadata as datetimes
                    opn = gdal.Open(j)
                    aq = [dt.datetime.strptime(opn.GetRasterBand(k).GetMetadata()['time'].split('.')[0],'%Y-%m-%dT%H:%M:%S')
                          for k in range(1,opn.RasterCount+1)]
                    
                    aquis_dates.append(aq)
                    
                    # get the resolution of the dataset
                    # on n==0, this is the blue band which is the highest resolution
                    res = opn.GetGeoTransform()[1]
                    if res < self.base_file_res:
                        # and set the attributes based off this file
                        if self.state_shapefile is not None:
                            self.base_gdal_obj = gdal.Warp('', j,  format='MEM',
                                            cutlineDSName=self.state_shapefile,
                                            cropToCutline=True,dstNodata = 0)
                        else:
                            self.base_gdal_obj = gdal.Open(j)
                            
                        self.base_file_res = res
                
                # fuse all the dates from all files and set them as attributes
                aquis_dates = np.concatenate(aquis_dates)
                self.dates = aquis_dates
            
            # add all the angle files to the list of files, so there are all the relevent 
            # paths in one place
            angle_files = np.concatenate([[j.replace(subproduct,subproduct+'_'+k) for j in monthly_files] for k in self.angle_keys])
            
            monthly_files = np.concatenate([monthly_files,angle_files])
            
            self.data_file_paths[i] = monthly_files    
    
    def _build_lookup(self):
        
        # in rare circumstances there are multiple observations on the same day
        # this trims the dates so the first observation in the date is used
        unique_dates, unique_inds = np.unique(self.dates,return_index=True)
  
        if len(unique_dates) != len(self.dates):
            self.dates = self.dates[unique_inds]
            for i in np.concatenate([['data'],self.angle_keys]):
                self.band_data[i] = self.band_data[i][:,unique_inds]
        
        # rebuild the lookup so the correct number of layers are added
        self.lookup = {key:n for n,(key,val) in enumerate(zip(self.dates,self.dates))}            
        self.bands_per_observation = {key: len(self.band_map) for key,val in zip(self.dates,self.dates)}
                    
                
    def _gather_data(self):
        
        # actually open all the data into memory
        
        # band_data is where all the data is stored, where it is indexed as:
        # [reflectance data or angle key][index for band][index for timestep layer]
        
        # self.band_data['data'] is a list of 3D arrays, where the first index
        # corresponds with the index for each of the self.band_map
        
        dict_keys = ['data']+self.angle_keys        
        self.band_data = {key: [] for key in dict_keys}
        
        # loop through each of the bands we are interested in
        for i in self.band_map:
            
            # find the list of files for this band
            flist = self.data_file_paths[i]       
            
            monthly_datasets = {key: [] for key in dict_keys}
            
            for j in flist:
                
                # find the repository key dataset belongs to
                
                # this is for angle datasets
                if True in [True for k in self.angle_keys if k in j]:
                    repo_key = [k for k in self.angle_keys if k in j][0]
                    
                    if i != self.band_map[0]:
                        continue
                # this is is for actual reflectance datasets
                else:
                    repo_key = 'data'
                    
                # find if the dataset needs reprojecting to the smallest resolution
                if self.__test_for_reporjection(j) == False:
                    # this means that it does NOT need reproecting and is at the
                    # histest resolution already
                    if self.state_shapefile is not None:
                        dataset = gdal.Warp('', j,  format='MEM',
                                    cutlineDSName=self.state_shapefile,
                                    cropToCutline=True,dstNodata = 0)
                    else:
                        dataset = gdal.Open(j)
                else:
                    # the dataset needs to be resampled as it is probably at 60m or 20m
                    dataset = self._reproject_image(j, clip_shapefile = self.state_shapefile)
                    # dataset is now at the right resolution

                # open the dataset as an array
                arr = dataset.ReadAsArray().astype(float)
                if np.ndim(arr) != 3:
                    arr = np.zeros([1,arr.shape[0],arr.shape[1]]) + arr
                # find the statemask attribute if it has not been done before
                if hasattr(self,'statemask') == False:
                    temporal_mean = np.nanmean(arr,axis=0)
                    safe_fmask = np.zeros([arr.shape[1],arr.shape[2]]).astype(bool)
                    safe_fmask[temporal_mean > 0] = True
                    self.statemask = safe_fmask
                
                # nan out the no data values                
                arr[arr == 0] = np.nan
                # for the unlikely event of a file only having 1 layer, it will be 2d which 
                # needs to changed so it fits in with the workflow
                # add the band values to a 3d array of the right shape
       
                # add it to the temporary repository
                monthly_datasets[repo_key].append(arr)
            
            # fuse the datasets together along the time axis and put them in the central datasets            
            for j in monthly_datasets:
                if len(monthly_datasets[j]) == 0:
                    continue
                self.band_data[j].append(np.concatenate(monthly_datasets[j]))
                
        for i in np.concatenate([['data'], self.angle_keys]):
            # turn everything into 4d arrays
            self.band_data[i] = np.array(self.band_data[i])
                
        # set the geotransform data
        proj = dataset.GetProjection()
        geoT = np.array(dataset.GetGeoTransform()).tolist()     
        self.output_info = [proj,geoT]
        
    def __test_for_reporjection(self,in_fname):
        
        # check if in_fname fits the right resolution of the lowest resolution of data
        tmp_opn = gdal.Open(in_fname)
        res = tmp_opn.GetGeoTransform()[1]
        if res == self.base_file_res:
            return False
        else:
            return True
        
    def _reproject_image(self,source_img, clip_shapefile = None, nodataval = 0):
        
        # use gdal to reporject the dataset to the lowest resolution 
        g = self.base_gdal_obj
        geo_t = g.GetGeoTransform()
        
        # get the size and extent of the blue band dataset
        x_size, y_size = g.RasterXSize, g.RasterYSize
        xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
        xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
        ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
        ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
        xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
        # get the projection and spatial reference of the data
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
        
        # use the shapefile to outline the field
        if clip_shapefile is not None:
            g = gdal.Warp('', source_img, format='MEM',
                      outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                      dstSRS=dstSRS, cutlineDSName=self.state_shapefile,
                      dstNodata = nodataval)
        # or dont, this option wont happen however as a shapefile will always 
        # be provided
        else:
            g = gdal.Warp('', source_img, format='MEM',
                      outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                      dstSRS=dstSRS, dstNodata = nodataval)
        return g


    def calc_lai(self, output_fname = None, inputs_smooth_factor = None):
        
        current_bands = np.copy(self.band_map)
        current_bands = ['B'+i.strip('0') for i in current_bands]
        ref = np.copy(self.band_data['data']) / 10000
        ref[ref==0] = np.nan
        if inputs_smooth_factor != None:
            ref = np.array([smoothn(y=i,s=inputs_smooth_factor,isrobust=True,axis=0)[0]
                  for i in ref])
            
        bd = {key: ref[n] for n,key in enumerate(current_bands)}
        bd['vza'] = self.band_data['vza'][0].astype(float)
        bd['sza'] = self.band_data['sza'][0].astype(float)
        
        lai_cube = LAI_evaluatePixelOrig(bd)
        
        if output_fname is not None:
            
            self._save_3d_data(lai_cube,
                              self.base_gdal_obj,
                              output_fname)
            
        else:
            
            return lai_cube
        
    def calc_fc(self, output_fname = None, inputs_smooth_factor = None):
        
        current_bands = np.copy(self.band_map)
        current_bands = ['B'+i.strip('0') for i in current_bands]
        ref = np.copy(self.band_data['data']) / 10000
        ref[ref==0] = np.nan
        if inputs_smooth_factor != None:
            ref = np.array([smoothn(y=i,s=inputs_smooth_factor,isrobust=True,axis=0)[0]
                  for i in ref])
            
        bd = {key: ref[n] for n,key in enumerate(current_bands)}
        bd['vza'] = self.band_data['vza'][0]
        bd['sza'] = self.band_data['sza'][0]
        bd['vaa'] = self.band_data['vaa'][0]
        bd['saa'] = self.band_data['saa'][0]
        
        fc_cube = FC_evaluatePixel(bd)
        
        if output_fname is not None:
            
            self._save_3d_data(fc_cube,
                              self.base_gdal_obj,
                              output_fname)
            
        else:
            
            return fc_cube
        
    def calc_fapar(self, output_fname = None, inputs_smooth_factor = None):
        
        current_bands = np.copy(self.band_map)
        current_bands = ['B'+i.strip('0') for i in current_bands]
        ref = np.copy(self.band_data['data']) / 10000
        ref[ref==0] = np.nan
        if inputs_smooth_factor != None:
            ref = np.array([smoothn(y=i,s=inputs_smooth_factor,isrobust=True,axis=0)[0]
                  for i in ref])
            
        bd = {key: ref[n] for n,key in enumerate(current_bands)}
        bd['vza'] = self.band_data['vza'][0]
        bd['sza'] = self.band_data['sza'][0]
        
        lai_cube = FAPAR_evaluatePixelOrig(bd)
        
        if output_fname is not None:
            
            self._save_3d_data(lai_cube,
                              self.base_gdal_obj,
                              output_fname)
            
        else:
            
            return lai_cube
        
    def calc_cab(self, output_fname = None, inputs_smooth_factor = None):
        
        current_bands = np.copy(self.band_map)
        current_bands = ['B'+i.strip('0') for i in current_bands]
        ref = np.copy(self.band_data['data']) / 10000
        ref[ref==0] = np.nan
        if inputs_smooth_factor != None:
            ref = np.array([smoothn(y=i,s=inputs_smooth_factor,isrobust=True,axis=0)[0]
                  for i in ref])
            
        bd = {key: ref[n] for n,key in enumerate(current_bands)}
        bd['vza'] = self.band_data['vza'][0]
        bd['sza'] = self.band_data['sza'][0]
        
        cab_cube = CAB_evaluatePixelOrig(bd)
        
        if output_fname is not None:
            
            self._save_3d_data(cab_cube,
                              self.base_gdal_obj,
                              output_fname)
            
        else:
            
            return lai_cube
            
    def _save_3d_data(self, in_arr, gdalobj, save_name):
    
        cols = in_arr.shape[2]
        rows = in_arr.shape[1]
        lyrcount = in_arr.shape[0]

        driver = gdal.GetDriverByName('GTiff')

        driver_options = ['COMPRESS=DEFLATE',
                              'BIGTIFF=YES',
                              'PREDICTOR=1',
                              'TILED=YES',
                              'COPY_SRC_OVERVIEWS=YES']

        outRaster = driver.Create(save_name, cols, rows, lyrcount, gdal.GDT_Float32, driver_options)
        outRaster.SetGeoTransform(gdalobj.GetGeoTransform())

        for n,i in enumerate(in_arr):
            outband = outRaster.GetRasterBand(n+1)
            outband.WriteArray(i)
            outband.SetMetadata({'aquisTime': self.dates[n].strftime('%Y-%m-%d_%H:%M:%S')})

        outRaster.SetProjection(gdalobj.GetProjection())
        outband.FlushCache()
        
    def calc_write_microcubes(self, parameter2write, tile_name, output_dir = None,
                             subproduct_alternate_name = None):
        
        """
        Method to calcualte the biophysical parameter and write it to microcubes.
        
        INPUTS:
             - parameter2write (string) - the name of the parameter to process e.g. lai
             - tile_name (string) - the name of the tile to write to
        OPTIONS:
             - output_dir (string) - where to write the data to. If left None, then it is 
                 written to the same datacube directory the S2 is opened from.
                 
        example:
        
        obj = MLEO_operator('/data/acropalis/datacube/',
                            '/data/acropalis/SHPs/adas_all_rpa/F0073_F0073-008.shp')
        tile = 'F0073_F0073-008'
        obj.calc_write_microcubes('lai', tile, output_dir='/data/acropalis/datacube/')
        """
        
        
        if output_dir is None:
            output_dir = self.tile_data_path
        
        implemeted = ['lai','fapar','fcover']
        if parameter2write not in implemeted:
            raise NotImplementedError('Pasrameter %s not currently implemeted'%parameter2write)
        
        methdict = {'lai': self.calc_lai,
                   'fapar': self.calc_fapar,
                   'fcover': self.calc_fc,
                   'cab': self.calc_cab}
        
        # calculate the parameter 
        dc = methdict[parameter2write]()
        
        # use xarray to get all the ancillary information 
        if xr.__version__ >= '0.20.0':
            # newer versions of xarray and rioxarray
            template = rioxarray.open_rasterio(self.data_file_paths['02'][0])
            template.attrs['crs'] = template.spatial_ref.attrs['spatial_ref']
        else:
            # older xarray handling
            template = xr.open_rasterio(self.data_file_paths['02'][0])
        
        # get the information about the arrays size and shape
        geot = self.base_gdal_obj.GetGeoTransform()        
        xsize = self.base_gdal_obj.RasterXSize
        ysize = self.base_gdal_obj.RasterYSize
        
        # xarray wants the the center of the pixels, so offset the corners by
        # half the pixel size
        xs = [geot[0]+(geot[1]*i) + (geot[1]/2) for i in np.arange(xsize)]
        ys = [geot[3]+(geot[5]*i) + (geot[5]/2) for i in np.arange(ysize)]
        
        # put everything into an xarray
        outds = xr.DataArray( dc,coords={'time': self.dates,
                                  'latitude': ys,
                                  'longitude': xs}, dims = ['time', 'latitude', 'longitude'])
        for i in template.attrs:
            if i != 'transform':
                outds.attrs[i] = template.attrs[i]
            else:
                outds.attrs[i] = geot
        
        # find the months and years in the dataset
        months = np.array([i.astype('datetime64[M]') for i in outds['time'].values])
        years = np.array([i.astype('datetime64[Y]') for i in outds['time'].values])
        
        attrs = outds.attrs
        
        # get the names 
        product = 'MLEO'
        if subproduct_alternate_name is None:
            subproduct = parameter2write
        else:
            subproduct = subproduct_alternate_name
            
        fill_value = 999
        
        # loop through the years and months to write the monthly datasets
        for year in np.unique(years):
            for month in np.unique(months):
                
                # find the indexes of the dataset
                ind = np.where((months == month) & (years == year))[0]
                fname_month = (str(month).split('-')[1])
                fname_year = (str(month).split('-')[0])
                
                fname_ext = f'{product}_{subproduct}_{tile_name}_{fname_year}-{fname_month}.tif'
                dataDir = os.path.join(output_dir,product,subproduct,tile_name)
                fname = os.path.join(dataDir, fname_ext)
            
                ds_monthly = outds.isel({'time': ind})
                
                # skip if the dataset is empty
                if len(ds_monthly['time'].values) == 0:
                    continue
                
                if os.path.isdir(dataDir) == False:
                    os.makedirs(dataDir)

                # Check if layer is empty
                layers_to_keep = []
                for _band in range(ds_monthly.shape[0]):
                    if not ds_monthly[_band].data.mean() == 0:
                        layers_to_keep.append(_band)

                ds_monthly = ds_monthly[layers_to_keep].to_dataset(name='GRD')

                # the above command does not retain the atributes of the dataset. So you
                # have to reset the attributes to the datarray
                ds_monthly.attrs = copy.deepcopy(attrs)
                #  Create metadata dictionary
                ds_md = {'add_offset': 0, 
                         'scale_factor': 1,
                         'fill_value': fill_value,
                         'version': 'Sentinel2 ToolBox Level2 Products',
                         'product': product,
                         'ATBD': 'https://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf'
                        }
                
                # save it
                save_xarray(fname, xarray=ds_monthly, data_var='GRD',metadata=ds_md)


###########################################################################
#    Generic Functions
###########################################################################

def normalize(unnormalized, mn, mx):
      return 2 * (unnormalized - mn) / (mx - mn) - 1

def denormalize(normalized, mn, mx):
      return 0.5 * (normalized + 1) * (mx - mn) + mn

def tansig(input_):
    return 2 / (1 + np.exp(-2 * input_)) - 1

# def degToRad():
#     return np.pi / 180

degToRad = np.pi / 180

###########################################################################
#    LAI
###########################################################################

def LAI_evaluatePixelOrig(in_bd):
    
    global degToRad
    
    b03_norm = normalize(in_bd['B3'], 0, 0.253061520471542)
    b04_norm = normalize(in_bd['B4'], 0, 0.290393577911328)
    b05_norm = normalize(in_bd['B5'], 0, 0.305398915248555)
    b06_norm = normalize(in_bd['B6'], 0.006637972542253, 0.608900395797889)
    b07_norm = normalize(in_bd['B7'], 0.013972727018939, 0.753827384322927)
    b8a_norm = normalize(in_bd['B8A'], 0.026690138082061, 0.782011770669178)
    b11_norm = normalize(in_bd['B11'], 0.016388074192258, 0.493761397883092)
    b12_norm = normalize(in_bd['B12'], 0, 0.493025984460231)
    viewZen_norm = normalize(np.cos(in_bd['vza'] * degToRad), 0.918595400582046, 1)
    sunZen_norm  = normalize(np.cos(in_bd['sza'] * degToRad), 0.342022871159208, 0.936206429175402)
    relAzim_norm = np.cos((in_bd['sza'] - in_bd['vza']) * degToRad)
    
    degToRad = np.pi / 180

    n1 = LAI_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,\
                 b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    
    n2 = LAI_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,\
                 b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    
    n3 = LAI_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,\
                 b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    
    n4 = LAI_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,\
                 b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    
    n5 = LAI_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,\
                 b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    
    l2 = LAI_layer2(n1, n2, n3, n4, n5)
    
    lai = denormalize(l2, 0.000319182538301, 14.4675094548151)
    
    return lai

def LAI_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,
            b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sm = 4.96238030555279 \
        - 0.023406878966470 * b03_norm \
        + 0.921655164636366 * b04_norm \
        + 0.135576544080099 * b05_norm \
        - 1.938331472397950 * b06_norm \
        - 3.342495816122680 * b07_norm \
        + 0.902277648009576 * b8a_norm \
        + 0.205363538258614 * b11_norm \
        - 0.040607844721716 * b12_norm \
        - 0.083196409727092 * viewZen_norm \
        + 0.260029270773809 * sunZen_norm \
        + 0.284761567218845 * relAzim_norm

    return tansig(sm)
    
def LAI_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,
             b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm = 1.416008443981500 \
        - 0.132555480856684 * b03_norm \
        - 0.139574837333540 * b04_norm \
        - 1.014606016898920 * b05_norm \
        - 1.330890038649270 * b06_norm \
        + 0.031730624503341 * b07_norm \
        - 1.433583541317050 * b8a_norm \
        - 0.959637898574699 * b11_norm \
        + 1.133115706551000 * b12_norm \
        + 0.216603876541632 * viewZen_norm \
        + 0.410652303762839 * sunZen_norm \
        + 0.064760155543506 * relAzim_norm

    return tansig(sm)

def LAI_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,
            b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm = 1.075897047213310 \
        + 0.086015977724868 * b03_norm \
        + 0.616648776881434 * b04_norm \
        + 0.678003876446556 * b05_norm \
        + 0.141102398644968 * b06_norm \
        - 0.096682206883546 * b07_norm \
        - 1.128832638862200 * b8a_norm \
        + 0.302189102741375 * b11_norm \
        + 0.434494937299725 * b12_norm \
        - 0.021903699490589 * viewZen_norm \
        - 0.228492476802263 * sunZen_norm \
        - 0.039460537589826 * relAzim_norm

    return tansig(sm)

def LAI_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,
            b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm =  1.533988264655420 \
        - 0.109366593670404 * b03_norm \
        - 0.071046262972729 * b04_norm \
        + 0.064582411478320 * b05_norm \
        + 2.906325236823160 * b06_norm \
        - 0.673873108979163 * b07_norm \
        - 3.838051868280840 * b8a_norm \
        + 1.695979344531530 * b11_norm \
        + 0.046950296081713 * b12_norm \
        - 0.049709652688365 * viewZen_norm \
        + 0.021829545430994 * sunZen_norm \
        + 0.057483827104091 * relAzim_norm

    return tansig(sm)

def LAI_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,
            b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sm = 3.024115930757230 \
        - 0.089939416159969 * b03_norm \
        + 0.175395483106147 * b04_norm \
        - 0.081847329172620 * b05_norm \
        + 2.219895367487790 * b06_norm \
        + 1.713873975136850 * b07_norm \
        + 0.713069186099534 * b8a_norm \
        + 0.138970813499201 * b11_norm \
        - 0.060771761518025 * b12_norm \
        + 0.124263341255473 * viewZen_norm \
        + 0.210086140404351 * sunZen_norm \
        - 0.183878138700341 * relAzim_norm

    return tansig(sm)

def LAI_layer2(neuron1, neuron2, neuron3, neuron4, neuron5):
    
    sm = 1.096963107077220 \
        - 1.500135489728730 * neuron1 \
        - 0.096283269121503 * neuron2 \
        - 0.194935930577094 * neuron3 \
        - 0.352305895755591 * neuron4 \
        + 0.075107415847473 * neuron5 \

    return sm

###########################################################################
#    FaPAR
###########################################################################


def FAPAR_evaluatePixelOrig(sample):
    
    global degToRad
    
    b03_norm = normalize(sample['B3'] , 0, 0.253061520471542)
    b04_norm = normalize(sample['B4'], 0, 0.290393577911328)
    b05_norm = normalize(sample['B5'], 0, 0.305398915248555)
    b06_norm = normalize(sample['B6'], 0.006637972542253, 0.608900395797889)
    b07_norm = normalize(sample['B7'], 0.013972727018939, 0.753827384322927)
    b8a_norm = normalize(sample['B8A'], 0.026690138082061, 0.782011770669178)
    b11_norm = normalize(sample['B11'], 0.016388074192258, 0.493761397883092)
    b12_norm = normalize(sample['B12'], 0, 0.493025984460231)
    viewZen_norm = normalize(np.cos(sample['vza'] * degToRad), 0.918595400582046, 1)
    sunZen_norm  = normalize(np.cos(sample['sza'] * degToRad), 0.342022871159208, 0.936206429175402)
    relAzim_norm = np.cos((sample['sza'] - sample['vza']) * degToRad)
  
    n1 = FAPAR_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                       b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n2 = FAPAR_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                       b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n3 = FAPAR_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                       b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n4 = FAPAR_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                       b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n5 = FAPAR_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                       b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    
    l2 = FAPAR_layer2(n1, n2, n3, n4, n5)
    
    fapar = denormalize(l2, 0.000153013463222, 0.977135096979553)
    return fapar

def FAPAR_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sum =\
    - 0.887068364040280\
    + 0.268714454733421 * b03_norm\
    - 0.205473108029835 * b04_norm\
    + 0.281765694196018 * b05_norm\
    + 1.337443412255980 * b06_norm\
    + 0.390319212938497 * b07_norm\
    - 3.612714342203350 * b8a_norm\
    + 0.222530960987244 * b11_norm\
    + 0.821790549667255 * b12_norm\
    - 0.093664567310731 * viewZen_norm\
    + 0.019290146147447 * sunZen_norm\
    + 0.037364446377188 * relAzim_norm

    return tansig(sum)

def FAPAR_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sum =\
    + 0.320126471197199\
    - 0.248998054599707 * b03_norm\
    - 0.571461305473124 * b04_norm\
    - 0.369957603466673 * b05_norm\
    + 0.246031694650909 * b06_norm\
    + 0.332536215252841 * b07_norm\
    + 0.438269896208887 * b8a_norm\
    + 0.819000551890450 * b11_norm\
    - 0.934931499059310 * b12_norm\
    + 0.082716247651866 * viewZen_norm\
    - 0.286978634108328 * sunZen_norm\
    - 0.035890968351662 * relAzim_norm

    return tansig(sum)

def FAPAR_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sum =\
    + 0.610523702500117\
    - 0.164063575315880 * b03_norm\
    - 0.126303285737763 * b04_norm\
    - 0.253670784366822 * b05_norm\
    - 0.321162835049381 * b06_norm\
    + 0.067082287973580 * b07_norm\
    + 2.029832288655260 * b8a_norm\
    - 0.023141228827722 * b11_norm\
    - 0.553176625657559 * b12_norm\
    + 0.059285451897783 * viewZen_norm\
    - 0.034334454541432 * sunZen_norm\
    - 0.031776704097009 * relAzim_norm

    return tansig(sum)

def FAPAR_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sum =\
    - 0.379156190833946\
    + 0.130240753003835 * b03_norm\
    + 0.236781035723321 * b04_norm\
    + 0.131811664093253 * b05_norm\
    - 0.250181799267664 * b06_norm\
    - 0.011364149953286 * b07_norm\
    - 1.857573214633520 * b8a_norm\
    - 0.146860751013916 * b11_norm\
    + 0.528008831372352 * b12_norm\
    - 0.046230769098303 * viewZen_norm\
    - 0.034509608392235 * sunZen_norm\
    + 0.031884395036004 * relAzim_norm

    return tansig(sum)

def FAPAR_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sum =\
    + 1.353023396690570\
    - 0.029929946166941 * b03_norm\
    + 0.795804414040809 * b04_norm\
    + 0.348025317624568 * b05_norm\
    + 0.943567007518504 * b06_norm\
    - 0.276341670431501 * b07_norm\
    - 2.946594180142590 * b8a_norm\
    + 0.289483073507500 * b11_norm\
    + 1.044006950440180 * b12_norm\
    - 0.000413031960419 * viewZen_norm\
    + 0.403331114840215 * sunZen_norm\
    + 0.068427130526696 * relAzim_norm

    return tansig(sum)

def FAPAR_layer2(neuron1, neuron2, neuron3, neuron4, neuron5):
    sum =\
    - 0.336431283973339\
    + 2.126038811064490 * neuron1\
    - 0.632044932794919 * neuron2\
    + 5.598995787206250 * neuron3\
    + 1.770444140578970 * neuron4\
    - 0.267879583604849 * neuron5

    return sum


###########################################################################
#    Fraction Cover
###########################################################################


def FC_evaluatePixel(in_bd):
    b03_norm = normalize(in_bd['B3'], 0, 0.253061520472)
    b04_norm = normalize(in_bd['B4'], 0, 0.290393577911)
    b05_norm = normalize(in_bd['B5'], 0, 0.305398915249)
    b06_norm = normalize(in_bd['B6'], 0.00663797254225, 0.608900395798)
    b07_norm = normalize(in_bd['B7'], 0.0139727270189, 0.753827384323)
    b8a_norm = normalize(in_bd['B8A'], 0.0266901380821, 0.782011770669)
    b11_norm = normalize(in_bd['B11'], 0.0163880741923, 0.493761397883)
    b12_norm = normalize(in_bd['B12'], 0, 0.49302598446)
    viewZen_norm = normalize(np.cos(in_bd['vza'] *degToRad), 0.918595400582, 0.999999999991)
    sunZen_norm  = normalize(np.cos(in_bd['sza'] * degToRad), 0.342022871159, 0.936206429175)
    relAzim_norm = np.cos((in_bd['saa']- in_bd['vaa']) * degToRad)
    
    n1 = FC_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
                    viewZen_norm,sunZen_norm,relAzim_norm)
    n2 = FC_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
                    viewZen_norm,sunZen_norm,relAzim_norm)
    n3 = FC_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
                    viewZen_norm,sunZen_norm,relAzim_norm)
    n4 = FC_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
                    viewZen_norm,sunZen_norm,relAzim_norm)
    n5 = FC_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
                    viewZen_norm,sunZen_norm,relAzim_norm)

    l2 = FC_layer2(n1, n2, n3, n4, n5)

    fcover = denormalize(l2, 0.000181230723879, 0.999638214715)
    
    return fcover


def FC_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
            viewZen_norm,sunZen_norm,relAzim_norm):
    sm = - 1.45261652206 \
        - 0.156854264841 * b03_norm \
        + 0.124234528462 * b04_norm \
        + 0.235625516229 * b05_norm \
        - 1.8323910258 * b06_norm \
        - 0.217188969888 * b07_norm \
        + 5.06933958064 * b8a_norm \
        - 0.887578008155 * b11_norm \
        - 1.0808468167 * b12_norm \
        - 0.0323167041864 * viewZen_norm \
        - 0.224476137359 * sunZen_norm \
        - 0.195523962947 * relAzim_norm

    return tansig(sm)

def FC_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
            viewZen_norm,sunZen_norm,relAzim_norm):
    sm = - 1.70417477557 \
        - 0.220824927842 * b03_norm \
        + 1.28595395487 * b04_norm \
        + 0.703139486363 * b05_norm \
        - 1.34481216665 * b06_norm \
        - 1.96881267559 * b07_norm \
        - 1.45444681639 * b8a_norm \
        + 1.02737560043 * b11_norm \
        - 0.12494641532 * b12_norm \
        + 0.0802762437265 * viewZen_norm \
        - 0.198705918577 * sunZen_norm \
        + 0.108527100527 * relAzim_norm

    return tansig(sm)

def FC_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm,
            viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm = + 1.02168965849 \
        - 0.409688743281 * b03_norm \
        + 1.08858884766 * b04_norm \
        + 0.36284522554 * b05_norm \
        + 0.0369390509705 * b06_norm \
        - 0.348012590003 * b07_norm \
        - 2.0035261881 * b8a_norm \
        + 0.0410357601757 * b11_norm \
        + 1.22373853174 * b12_norm \
        + -0.0124082778287 * viewZen_norm \
        - 0.282223364524 * sunZen_norm \
        + 0.0994993117557 * relAzim_norm
    
    return tansig(sm)

def FC_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
            viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm = - 0.498002810205 \
        - 0.188970957866 * b03_norm \
        - 0.0358621840833 * b04_norm \
        + 0.00551248528107 * b05_norm \
        + 1.35391570802 * b06_norm \
        - 0.739689896116 * b07_norm \
        - 2.21719530107 * b8a_norm \
        + 0.313216124198 * b11_norm \
        + 1.5020168915 * b12_norm \
        + 1.21530490195 * viewZen_norm \
        - 0.421938358618 * sunZen_norm \
        + 1.48852484547 * relAzim_norm \
    
    return tansig(sm)

def FC_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, 
            viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm = - 3.88922154789 \
        + 2.49293993709 * b03_norm \
        - 4.40511331388 * b04_norm \
        - 1.91062012624 * b05_norm \
        - 0.703174115575 * b06_norm \
        - 0.215104721138 * b07_norm \
        - 0.972151494818 * b8a_norm \
        - 0.930752241278 * b11_norm \
        + 1.2143441876 * b12_norm \
        - 0.521665460192 * viewZen_norm \
        - 0.445755955598 * sunZen_norm \
        + 0.344111873777 * relAzim_norm

    return tansig(sm)

def FC_layer2(neuron1, neuron2, neuron3, neuron4, neuron5):
    sm = - 0.0967998147811 \
        + 0.23080586765 * neuron1 \
        - 0.333655484884 * neuron2 \
        - 0.499418292325 * neuron3 \
        + 0.0472484396749 * neuron4 \
        - 0.0798516540739 * neuron5 \
        
    return sm

###########################################################################
#    CAB leaf content
###########################################################################

def CAB_evaluatePixel(in_bd):

    b03_norm = normalize(in_bd['B3'], 0, 0.253061520471542)
    b04_norm = normalize(in_bd['B4'], 0, 0.290393577911328)
    b05_norm = normalize(in_bd['B5'], 0, 0.305398915248555)
    b06_norm = normalize(in_bd['B6'], 0.006637972542253, 0.608900395797889)
    b07_norm = normalize(in_bd['B7'], 0.013972727018939, 0.753827384322927)
    b8a_norm = normalize(in_bd['B8A'], 0.026690138082061, 0.782011770669178)
    b11_norm = normalize(in_bd['B11'], 0.016388074192258, 0.493761397883092)
    b12_norm = normalize(in_bd['B12'], 0, 0.493025984460231)
    viewZen_norm = normalize(np.cos(in_bd['vza'] * degToRad), 0.918595400582046, 1)
    sunZen_norm  = normalize(np.cos(in_bd['sza'] * degToRad), 0.342022871159208, 0.936206429175402)
    relAzim_norm = np.cos((in_bd['saa'] - in_bd['vaa']) * degToRad)

    n1 = CAB_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n2 = CAB_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n3 = CAB_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n4 = CAB_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n5 = CAB_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)

    l2 = CAB_layer2(n1, n2, n3, n4, n5)

    cab = denormalize(l2, 0.007426692959872, 873.908222110306) / 300
    
    return cab

def CAB_neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
     
    sm = 4.242299670155190 \
          + 0.400396555256580 * b03_norm \
          + 0.607936279259404 * b04_norm \
          + 0.137468650780226 * b05_norm \
          - 2.955866573461640 * b06_norm \
          - 3.186746687729570 * b07_norm \
          + 2.206800751246430 * b8a_norm \
          - 0.313784336139636 * b11_norm \
          + 0.256063547510639 * b12_norm \
          - 0.071613219805105 * viewZen_norm \
          + 0.510113504210111 * sunZen_norm \
          + 0.142813982138661 * relAzim_norm

    return tansig(sm)

def CAB_neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    
     sm = - 0.259569088225796 \
            - 0.250781102414872 * b03_norm \
            + 0.439086302920381 * b04_norm \
            - 1.160590937522300 * b05_norm \
            - 1.861935250269610 * b06_norm \
            + 0.981359868451638 * b07_norm \
            + 1.634230834254840 * b8a_norm \
            - 0.872527934645577 * b11_norm \
            + 0.448240475035072 * b12_norm \
            + 0.037078083501217 * viewZen_norm \
            + 0.030044189670404 * sunZen_norm \
            + 0.005956686619403 * relAzim_norm \

     return tansig(sm)
    
def CAB_neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm =  3.130392627338360 \
            + 0.552080132568747 * b03_norm \
            - 0.502919673166901 * b04_norm \
            + 6.105041924966230 * b05_norm \
            - 1.294386119140800 * b06_norm \
            - 1.059956388352800 * b07_norm \
            - 1.394092902418820 * b8a_norm \
            + 0.324752732710706 * b11_norm \
            - 1.758871822827680 * b12_norm \
            - 0.036663679860328 * viewZen_norm \
            - 0.183105291400739 * sunZen_norm \
            - 0.038145312117381 * relAzim_norm

    return tansig(sm)

def CAB_neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    
    sm = + 0.774423577181620 \
            + 0.211591184882422 * b03_norm \
            - 0.248788896074327 * b04_norm \
            + 0.887151598039092 * b05_norm \
            + 1.143675895571410 * b06_norm \
            - 0.753968830338323 * b07_norm \
            - 1.185456953076760 * b8a_norm \
            + 0.541897860471577 * b11_norm \
            - 0.252685834607768 * b12_norm \
            - 0.023414901078143 * viewZen_norm \
            - 0.046022503549557 * sunZen_norm \
            - 0.006570284080657 * relAzim_norm \

    return tansig(sm)

def CAB_neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    sm = 2.584276648534610 \
            + 0.254790234231378 * b03_norm \
            - 0.724968611431065 * b04_norm \
            + 0.731872806026834 * b05_norm \
            + 2.303453821021270 * b06_norm \
            - 0.849907966921912 * b07_norm \
            - 6.425315500537270 * b8a_norm \
            + 2.238844558459030 * b11_norm \
            - 0.199937574297990 * b12_norm \
            + 0.097303331714567 * viewZen_norm \
            + 0.334528254938326 * sunZen_norm \
            + 0.113075306591838 * relAzim_norm

    return tansig(sm)

def CAB_layer2(neuron1, neuron2, neuron3, neuron4, neuron5):
    sm = 0.463426463933822 \
            - 0.352760040599190 * neuron1 \
            - 0.603407399151276 * neuron2 \
            + 0.135099379384275 * neuron3 \
            - 1.735673123851930 * neuron4 \
            - 0.147546813318256 * neuron5 

    return sm


