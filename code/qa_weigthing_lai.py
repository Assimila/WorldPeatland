
from glob import glob
import osgeo.gdal as gdal
import rasterio
import xarray as xr
import rioxarray

site = 'Degero'
product = 'MCD15A3H.061'
variable = 'Lai_500m'            # 'Fpar_500m'           # 'Lai_500m'
uncert_var = 'LaiStdDev_500m'    # 'FparStdDev_500m'     # 'LaiStdDev_500m'
scaling_factor = 0.1             # 0.01                  # 0.1
weighting_factor = 0.2  # Define the weighting factor when the obs are interpolated/smoothed

# Paths to the VRT file and mask file
lai_file = f"/wp_data/sites/{site}/MODIS/{product}/h??v??/{variable}/interpolated/{product}._{variable}.linear.smoothn.1.5.tif"
lai_file = glob(lai_file)[0] 

lai_stddev_file = f"/wp_data/sites/{site}/MODIS/{product}/h??v??/{uncert_var}/{uncert_var}.vrt"
lai_stddev_file = glob(lai_stddev_file)[0]

mask_file = f"/wp_data/sites/{site}/MODIS/analytics_QA_settings/_{variable}_qa_analytics_mask.tif"
mask_file = glob(mask_file)[0]

# Open the VRT file using rioxarray
lai = rioxarray.open_rasterio(lai_file, mask_and_scale=True).astype("float32") * scaling_factor

# Open the LaiStdDev VRT file using rioxarray
lai_stddev = rioxarray.open_rasterio(lai_stddev_file)
# Mask lai_stddev where values are between 248 and 255
# https://lpdaac.usgs.gov/documents/926/MOD15_User_Guide_V61.pdf Table 7
lai_stddev = xr.where((lai_stddev >= 248) & (lai_stddev <= 255), 0, lai_stddev).astype("float32") * scaling_factor

# Open the mask file using rioxarray
mask = rioxarray.open_rasterio(mask_file)

# Check if both layers have the same shape
if lai.shape == mask.shape:
    print("The data and mask have the same shape.")
else:
    print("The data and mask do not have the same shape.")
    print(f"Data shape: {lai.shape}")
    print(f"Mask shape: {mask.shape}")
    raise ValueError("Data and mask shapes do not match.")


# Apply the weighting factor where the mask is 0, keep original values where mask is 1
lai_stddev_weighted = xr.where(mask == 0, lai * weighting_factor, lai_stddev)

# Add the new weighted LAI stddev as a new variable in the dataset
xarray_data = lai.to_dataset(name=variable)
xarray_data[uncert_var] = lai_stddev_weighted

# Extract per-band tags using rasterio
with rasterio.open(lai_file) as src:
    tags_per_band = [src.tags(i) for i in range(1, src.count + 1)]
    geotransform = src.transform.to_gdal()
    projection = src.crs.to_wkt()

bands, height, width = xarray_data[uncert_var].data.shape

# Save the updated dataset to a Cloud Optimized GeoTIFF
fname = lai_file.replace(".tif", "_qa_weighted_std_dev.tif")
print(fname)

# Create output with GDAL
driver = gdal.GetDriverByName("GTiff")
driver_options = ['COMPRESS=DEFLATE',
                  'BIGTIFF=YES',
                  'PREDICTOR=1',
                  'TILED=YES',
                  'COPY_SRC_OVERVIEWS=YES']

out_ds = driver.Create(fname, width, height, bands, gdal.GDT_Float32,
    options=driver_options
)

out_ds.SetGeoTransform(geotransform)
out_ds.SetProjection(projection)

# Write each band and set per-band metadata
for i in range(bands):
    out_band = out_ds.GetRasterBand(i + 1)
    out_band.WriteArray(xarray_data[uncert_var].data[i, :, :])
    # Set per-band tags
    for k, v in tags_per_band[i].items():
        out_band.SetMetadataItem(k, v)
    out_band.FlushCache()

out_ds.FlushCache()
out_ds = None

# xarray_data[uncert_var].rio.to_raster(
#     fname, 
#     driver="GTiff", 
#     compress="DEFLATE",
#     tags_per_band=tags_per_band
# )

print("Saved the weighted LAI stddev as a GeoTIFF.")
