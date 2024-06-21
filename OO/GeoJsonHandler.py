import os
from osgeo import ogr, osr
import logging
from jsonschema import validate
import json 

LOG = logging.getLogger()

class GeoJsonHandler:
    def __init__(self, geojson_path):
        
        # storing input geojson path as attribute 
        self.geojson_path = geojson_path
        # private method for this class
        self.__sanity_check()
        
        # setting site_name, country
        self.set_str_attributes()
        # setting _extent_wgs84
        self.set_extent_wgs84()
        # set the polygon bbox to get MODIS tiles 
        set_polygon_bbox()
        # transform extent coordinate system into sinusoidal
        self.get_extent_sinusoidal() 

    @classmethod
    def __sanity_check(self):
        
        ''' 
        check that attributes are available
        TODO use the Json schema https://builtin.com/software-engineering-perspectives/python-json-schema
        '''
        if not os.path.isfile(self.geojson_path):
            LOG.error('GeoJSON file does not exist')
            return None
        
        # load the schema json file 
        schema_data = open('Demo.json')
        schema = json.load(schema_data)
        # load the input geojson file path
        json_object = open(self.geojson_path)
        self.jsonData = json.load(json_object)

        validate(
            instance=self.jsonData,
            schema=schema,
        )
    
    @classmethod
    def set_str_attributes(self):
        
        ''' Set all string attributes from user input jsonData'''
        
        # TODO set in documentation a template of config 
        # list of available countries for VIIRS add to documentation
        
        feat = self.jsonData['features'][0]['properties']
        if 'site_area' in feat:
            self.site_area = feat['site_area']
        if 'country' in feat:
            self.country = feat['country'] 

    @classmethod
    def set_extent_wgs84(self):
        '''
        Extract GeoJson features, bbox layer of a GeoJson file site and gives an osgeo geometry object.

        INPUT:
            - geojson_path (str): Path/location of the GeoJson file of the site (given by user).

        OUTPUT:
            - _extent (tuple): Bounding box coordinates (minX, maxX, minY, maxY).
            - site_name (str): Name of the site area from GeoJson file.
            - country (str): Country where the site is located.
        '''
        driver = ogr.GetDriverByName("GeoJSON")
        src_GeoJSON = driver.Open(self.geojson_path)

        if src_GeoJSON is None:
            LOG.error("Error opening GeoJSON file")

        try:
            site_layer = src_GeoJSON.GetLayer()

            if site_layer.GetGeomType() != ogr.wkbPolygon:
                raise Exception("The GeoJSON geometry is not a polygon")
            
            # extract the bbox from the json layer
            self._extent_wgs84 = site_layer.GetExtent()
            
        except Exception as e:
            LOG.error(f'Error extracting attributes from GeoJSON file: {e}')
            return False 
        finally:
            src_GeoJSON = None
            return True
        
    @classmethod
    def set_polygon_bbox(self):
        
        '''
        This function gets a polygon osgeo object used later on to get the MODIS tiles
        
        OUTPUT
            polygon (osgeo.ogr.Geometry) - polygon geometry object containing extent lat and lon of the site
        '''
        
        min_x, max_x, min_y, max_y = self._extent
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(min_x, min_y)
        ring.AddPoint(max_x, min_y)
        ring.AddPoint(max_x, max_y)
        ring.AddPoint(min_x, max_y)
        ring.AddPoint(min_x, min_y)
        self.polygon = ogr.Geometry(ogr.wkbPolygon)
        self.polygon.AddGeometry(ring)
        
        return True
    
    @classmethod
    def get_extent_sinusoidal(self):
        
        '''
        This function changes the coordinate system of the extent of the input GeoJson 
        from WGS84 to sinusoidal using Proj4 string.

        OUTPUT:
            - extent (tuple): Bounding box coordinates (minX, maxX, minY, maxY) in sinusoidal.
        '''
        
        # TODO change the transformation str proj4 to using the code
        
        min_x, max_x, min_y, max_y = self._extent_wgs84
        
        # change coordinate system to sinusoidal used in ts_generator to crop tile to GeoJson geometry 
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
        
        self.extent_sinusoidal = (minX, maxX, minY, maxY)
        
        return True
