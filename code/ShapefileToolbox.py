try:
    import gdal
    import ogr
    import osr
except:
    from osgeo import gdal
    from osgeo import osr
    from osgeo import ogr
    from osgeo import gdal_array
    
import osgeo

import numpy as np
import json
import shapely

class ShapefileToolboxException(Exception):
    pass

class ShapefileToolbox():
    
    """
    ShapefileToolbox help:
    A module to easily perform simple transformations and other manipulations to a shapefile,
    where the shapefile can be in any original projection and outputs can be either in
    latitude/longitude (WGS84/EPSG:4326) or in meters (UTM/EPSG:326XX).
    
    Terminology:
        Perimeter - the outline of the boundary.
        Envelope - the smallest rectangle that exactly contains the perimeter.
        Centroid - the center of mass of the 2d polygon.
        
    EXAMPLE:
    
    import matplotlib.pyplot as plt
    
    # instantiate the toolbox    
    shpt = ShapefileToolbox('/data/NUE_Profits/shapefiles/NUEF004.shp')
    # first, get the centroid
    cent = shpt.get_wgs84_centroid()
    
    # next read the perimeter in both supported 
    # reference systems
    # buffering of the perimeter is supported, by changing the 
    # buffer_meter argument, all in meters. Default is 0.
    dists = [-20,30,0]

    fig,axs = plt.subplots(1,2, figsize=(6,3))
    axs[0].scatter(cent['lon'],cent['lat'])

    for i in dists:    
        wgs_perim = shpt.get_wgs84_perimeter(buffer_meters=i)
        utm_perim = shpt.get_utm_perimeter(buffer_meters=i)
        axs[0].plot(wgs_perim['lon'],wgs_perim['lat'],label='Buffer = %s'%i)
        axs[1].plot(utm_perim['lon'],utm_perim['lat'],label='Buffer = %s'%i)

    [i.legend() for i in axs]
    [i.grid() for i in axs]
    plt.tight_layout()
    
    # next get the envelope containing the perimeter
    # where the envelope is buffered by 50m
    wgs_env = shpt.get_wgs84_envelope(buffer_meters=50)
    utm_env = shpt.get_utm_envelope(buffer_meters=50)

    fig,axs = plt.subplots(1,2,figsize=(6,3))
    axs[0].plot(wgs_env['lon'], wgs_env['lat'])
    axs[0].plot(wgs_perim['lon'], wgs_perim['lat'])
    axs[1].plot(utm_env['lon'], utm_env['lat'])
    axs[1].plot(utm_perim['lon'], utm_perim['lat'])
    plt.tight_layout()
    
    # lastly, simplify the utm perimeter 
    utm_perim_s20 = shpt.simplify_utm_perimeter(20)
    utm_perim_s100 = shpt.simplify_utm_perimeter(100)

    plt.figure(figsize=(5,5))
    plt.plot(utm_perim['lon'], utm_perim['lat'],label='raw')
    plt.plot(utm_perim_s20['lon'], utm_perim_s20['lat'],label='s=20')
    plt.plot(utm_perim_s100['lon'], utm_perim_s100['lat'],label='s=100')
    plt.legend()
    
    # write some outputs
    shpt.write_output(wgs_perim, '/media/DataShare/Alex/data/shapefiles/testing_toolbox_wgs84_perimter.shp')
    shpt.write_output(cent, '/media/DataShare/Alex/data/shapefiles/testing_toolbox_centroid.shp')
    shpt.write_output(utm_env, '/media/DataShare/Alex/data/shapefiles/testing_toolbox_utm_env.shp')
    
    # write some outputs as geojsons
    shpt.write_output(wgs_perim, '/media/DataShare/Alex/data/shapefiles/testing_toolbox_wgs84_perimter.geojson',
                      output_driver='GeoJson')
    """
    
    def __init__(self, shapefile_path):
        
        self.shapefile_path = shapefile_path     
        self._run_inital_checks(self.shapefile_path)
        
        # find the epsg code of the centroid, for reference in UTM methods
        self.__get_epsg_code()
        
    def __get_epsg_code(self):
        
        wgs_cent = self.get_wgs84_centroid()
        epsg_code = self.__find_epsg_code(wgs_cent['geometry'])
        self.epsg_code = epsg_code 
        
    def _run_inital_checks(self, in_sfile):

        opn = ogr.Open(in_sfile)
        if opn is None:
            raise ShapefileToolboxException('The following shapefile is corrupted and not readable:\n%s'%in_sfile)

        lyr = opn.GetLayer()
        lcount = lyr.GetFeatureCount()
        if lcount == 0:
            raise ShapefileToolboxException('The following shapefile is empty and has no listed features:\n%s'%in_sfile)

        if lcount > 1:
            raise ShapefileToolboxException('The following shapefile is MultiFeature and not supported:\n%s'%in_sfile)

        feat = lyr.GetFeature(0)
        geom = feat.geometry()
        if geom is None:
            raise ShapefileToolboxException('The following shapefile has 1 feature but it has no coordinates:\n%s'%in_sfile)
            
        gtype = geom.GetGeometryName()
        if gtype != 'POLYGON':
            raise ShapefileToolboxException('The following shapefile is only has a MultiPolygon type which is not supported:\n%s'%in_sfile)

        opn = lyr = feat = geom = None
        
    def get_raw_perimeter(self):
        
        # open the dataset 
        shp = ogr.Open(self.shapefile_path)
        lyr = shp.GetLayer()
        # find first feature
        feat = lyr.GetFeature(0)
        geom = feat.geometry()
        # export the geomery as a json for easy handling
        coords = np.array(json.loads(geom.ExportToJson())['coordinates'][0])
        print (lyr)
        # package up the results
        return {'lat': coords[:,0], 'lon': coords[:,1],
               'srs': lyr.GetSpatialRef().ExportToWkt()}

    def get_wgs84_perimeter(self, buffer_meters = 0):
        
        # open the shapefile
        shp = ogr.Open(self.shapefile_path)
        lyr = shp.GetLayer()
        feat = lyr.GetFeature(0)
        geom = feat.geometry()
        
        # establish the transform
        current_srs = osr.SpatialReference()
        current_srs.ImportFromWkt(lyr.GetSpatialRef().ExportToWkt())
        if int(osgeo.__version__[0]) >= 3:
            current_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
        utm_srs = osr.SpatialReference()
        utm_srs.ImportFromEPSG(self.epsg_code)
        if int(osgeo.__version__[0]) >= 3:
            utm_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        current2utm = osr.CoordinateTransformation(current_srs,
                                                     utm_srs)
        geom.Transform(current2utm)
        
        coords = np.array(json.loads(geom.ExportToJson())['coordinates'][0])
        
        # create a shapely object to perform the buffering with
        shapely_ply = shapely.geometry.Polygon(coords)
        if buffer_meters != 0:
            shapely_ply = shapely_ply.buffer(buffer_meters, join_style=2)
        buffered_geom = ogr.CreateGeometryFromWkt(shapely_ply.wkt)
        
        # establish the output projection system
        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(4326)
        if int(osgeo.__version__[0]) >= 3:
            out_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
        # create and transform a utm feature so the output is in wgs84
        out_transform = osr.CoordinateTransformation(utm_srs,out_srs)
        buffered_geom.Transform(out_transform)
        
        # package the result
        env_coords = np.array(json.loads(buffered_geom.ExportToJson())['coordinates'][0])
        return {'lat': env_coords[:,1], 'lon': env_coords[:,0], 
                'srs': out_srs.ExportToWkt()}
    
    def get_utm_perimeter(self, buffer_meters = 0):
        
        # open the shapefile
        shp = ogr.Open(self.shapefile_path)
        lyr = shp.GetLayer()
        feat = lyr.GetFeature(0)
        geom = feat.geometry()
        
        # establish the transform
        current_srs = osr.SpatialReference()
        current_srs.ImportFromWkt(lyr.GetSpatialRef().ExportToWkt())
        if int(osgeo.__version__[0]) >= 3:
            current_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
        utm_srs = osr.SpatialReference()
        utm_srs.ImportFromEPSG(self.epsg_code)
        if int(osgeo.__version__[0]) >= 3:
            utm_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        current2utm = osr.CoordinateTransformation(current_srs,
                                                     utm_srs)
        geom.Transform(current2utm)
        
        coords = np.array(json.loads(geom.ExportToJson())['coordinates'][0])
        
        # create a shapely object to perform the buffering with
        shapely_ply = shapely.geometry.Polygon(coords)
        if buffer_meters != 0:
            shapely_ply = shapely_ply.buffer(buffer_meters, join_style=2)
        
        # no need to reproject the points
        return {'lat': shapely_ply.exterior.xy[1], 'lon': shapely_ply.exterior.xy[0], 
                'srs': utm_srs.ExportToWkt()}
            
    def get_wgs84_centroid(self):
        
        # open the shapefile
        shp = ogr.Open(self.shapefile_path)
        lyr = shp.GetLayer()
        feat = lyr.GetFeature(0)
        geom = feat.geometry()
        cent = geom.Centroid()
        
        # find the current projection system
        current_srs = osr.SpatialReference()
        current_srs.ImportFromWkt(lyr.GetSpatialRef().ExportToWkt())
        if int(osgeo.__version__[0]) >= 3:
            current_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
        # establish the target projection system        
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        if int(osgeo.__version__[0]) >= 3:
            wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        # write the transorm 
        current2wgs84 = osr.CoordinateTransformation(current_srs,
                                                     wgs84_srs)
        # and use it
        cent.Transform(current2wgs84)
        
        return {'lat': cent.GetY(), 'lon': cent.GetX(), 
                'srs': wgs84_srs.ExportToWkt(), 'geometry': cent}
    
    def __find_epsg_code(self, wgs_cent):
        
        # open the shapefile with all the different UTM zones contained
        path = '../utmzone-polygon/utmzone-polygon.shp'
        utm_shp = ogr.Open(path)
       
        utm_lyr = utm_shp.GetLayer()
        
        utm_zone = -9999
        
        # loop throguh each possible utm zones
        for fnum in range(utm_lyr.GetFeatureCount()):

            utm_feat = utm_lyr.GetFeature(fnum)
            utm_geom = utm_feat.geometry()

            utm_ref = utm_feat.GetGeometryRef()
            
            # if the centroid is within this feature, then break the loop and
            # declare the utm zone
            if utm_ref.Contains(wgs_cent) == True:
              
                utm_zone = utm_feat.GetField('Name')
                break
        
        if utm_zone == -9999:
            raise ValueError('UTM zone could not be established. Check the projection information.')
        
        # https://gis.stackexchange.com/questions/365584/convert-utm-zone-into-epsg-code
        target_epsg = 32600
        target_epsg += int(utm_zone)
        
        return target_epsg
    
    def get_wgs84_envelope(self, buffer_meters = 0):
        
        # open the shapefile
        shp = ogr.Open(self.shapefile_path)
        lyr = shp.GetLayer()
        feat = lyr.GetFeature(0)
        geom = feat.geometry()
        
        # establish the transform
        current_srs = osr.SpatialReference()
        current_srs.ImportFromWkt(lyr.GetSpatialRef().ExportToWkt())
        if int(osgeo.__version__[0]) >= 3:
            current_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
        utm_srs = osr.SpatialReference()
        utm_srs.ImportFromEPSG(self.epsg_code)
        if int(osgeo.__version__[0]) >= 3:
            utm_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        current2utm = osr.CoordinateTransformation(current_srs,
                                                     utm_srs)
        geom.Transform(current2utm)
        
        # find the encasing envelope
        env = geom.GetEnvelope()
        # pull out the corners         
        tl = [env[0],env[3]]
        tr = [env[1],env[3]]
        br = [env[1],env[2]]
        bl = [env[0],env[2]]
        
        # create a shapely object to perform the buffering with
        shapely_env = shapely.geometry.Polygon([bl, br, tr, tl])
        shapely_env_b = shapely_env.buffer(buffer_meters, join_style=2)
        buffered_geom = ogr.CreateGeometryFromWkt(shapely_env_b.wkt)
        
        # establish the output projection system
        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(4326)
        if int(osgeo.__version__[0]) >= 3:
            out_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
        # create and transform a utm feature so the output is in wgs84
        out_transform = osr.CoordinateTransformation(utm_srs,out_srs)
        buffered_geom.Transform(out_transform)
        
        # package the result
        env_coords = np.array(json.loads(buffered_geom.ExportToJson())['coordinates'][0])
        return {'lat': env_coords[:,1], 'lon': env_coords[:,0], 
                'srs': out_srs.ExportToWkt()}
    
    def get_utm_envelope(self, buffer_meters = 0):
        
        # open the shapefile
        shp = ogr.Open(self.shapefile_path)
        lyr = shp.GetLayer()
        feat = lyr.GetFeature(0)
        geom = feat.geometry()
        
        # establish the transform to meters
        current_srs = osr.SpatialReference()
        current_srs.ImportFromWkt(lyr.GetSpatialRef().ExportToWkt())
        if int(osgeo.__version__[0]) >= 3:
            current_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
        utm_srs = osr.SpatialReference()
        utm_srs.ImportFromEPSG(self.epsg_code)
        if int(osgeo.__version__[0]) >= 3:
            utm_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        current2utm = osr.CoordinateTransformation(current_srs,
                                                     utm_srs)
        geom.Transform(current2utm)
        # get the surrounding envelope
        env = geom.GetEnvelope()
        tl = [env[0],env[3]]
        tr = [env[1],env[3]]
        br = [env[1],env[2]]
        bl = [env[0],env[2]]
        # use shapely to buffer
        shapely_env = shapely.geometry.Polygon([bl, br, tr, tl])
        shapely_env_b = shapely_env.buffer(buffer_meters, join_style=2)
        # no need to transform anything        
        coords = shapely_env_b.exterior.xy
        
        return {'lat': coords[1], 'lon': coords[0],
                'srs': utm_srs.ExportToWkt()}
    
    def simplify_utm_perimeter(self, simplify_factor):
        
        # get the utm perimeter
        pdict = self.get_utm_perimeter()        
        # create a shapely object with his information
        ply = shapely.geometry.Polygon(zip(pdict['lon'], pdict['lat']))
        # simplify it, given the parameters specified
        ply_s = ply.simplify(simplify_factor)
        # export and wrap the results up
        coords = ply_s.exterior.xy
        return {'lat': coords[1], 'lon': coords[0], 'srs': pdict['srs']}
    
    def write_output(self, output, output_fname): 
    
        out_srs = osr.SpatialReference()
        out_srs.ImportFromWkt(output['srs'])

        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(output_fname)

        if type(output['lat']) == float:
            geom_type = ogr.wkbPoint
            output_shapely_geom = shapely.geometry.Point(output['lon'],
                                                          output['lat'])
            out_geom = ogr.CreateGeometryFromWkb(output_shapely_geom.wkb)
            
        else:
            geom_type = ogr.wkbPolygon
            output_shapely_geom = shapely.geometry.Polygon(zip(output['lon'],
                                                          output['lat']))
            out_geom = ogr.CreateGeometryFromWkb(output_shapely_geom.wkb)

        layer = ds.CreateLayer('', srs = out_srs, geom_type=geom_type)
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()
        new_feat = ogr.Feature(defn)
        new_feat.SetField('id', 1)
        new_feat.SetGeometry(out_geom)

        layer.CreateFeature(new_feat)
        feat = geom = ds = layer = feat = geom = None    
