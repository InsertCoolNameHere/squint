import gdal
from affine import Affine
import geohash2
import numpy as np
from osgeo import ogr
from osgeo import osr


def retrieve_pixel_value(x,y,data_source):
    #print(data_source.GetProjection())
    #print(data_source.GetGeoTransform())
    forward_transform = Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    pixel_coord = px, py

    return pixel_coord

#RETURNS LAT-LON BOUNDS OF A GEOHASH
def decode_geohash(geocode):
    lat, lon, l1, l2 = geohash2.decode_exactly(geocode)
    return lat-l1, lat+l1, lon-l2, lon+l2

def get_gdal_obj(filename):
    return gdal.Open(filename)

def get_pixel_from_lat_lon(latlons, dataset):
    #converting coordinate systems
    # Setup the source projection
    source = osr.SpatialReference()
    source.ImportFromWkt(dataset.GetProjection())
    #print(source)
    # The target projection
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    # Create the transform - this can be used repeatedly
    transform = osr.CoordinateTransformation(target, source)

    pixels=[]
    for lat,lon in latlons:
        x, y, z = transform.TransformPoint(lon, lat)
        #print("TRANSFORMED:", x, y, z)
        pixels.append(retrieve_pixel_value(x, y, dataset))
    return pixels

# CROP OUT A SMALLER GEOHASH FROM A LARGER GEOHASH TILE
def crop_geohash(geocode, datafile):
    lat1, lat2, lon1, lon2 = decode_geohash(geocode)
    return crop_section(lat1, lat2, lon1, lon2, datafile)

# CROPPING A RECTANGLE OUT OF AN IMAGE
def crop_section(lat1, lat2, lon1, lon2, datafile):
    latlons = []
    latlons.append((lat1, lon1))
    latlons.append((lat2, lon1))
    latlons.append((lat2, lon2))
    latlons.append((lat1, lon2))
    return get_pixel_from_lat_lon(latlons, datafile)


# ************************METHOD CALLS
geocode='9xjm5k'
print("DECODED:",geocode,decode_geohash(geocode))

filename = "/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/train/9xjm5_20190407.tif" #path to raster
lon,lat = -104.9635,40.2769
latlons = []
latlons.append((lat,lon))
print("PIXEL:",get_pixel_from_lat_lon(latlons,get_gdal_obj(filename)))

print("PIXEL:",crop_geohash(geocode,get_gdal_obj(filename)))
