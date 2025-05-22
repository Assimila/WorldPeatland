
from glob import glob
import xarray as xr
import rioxarray

site = 'Degero'
product = 'MCD15A3H.061'
variable = 'Fpar_500m'           # 'Lai_500m'
uncert_var = 'FparStdDev_500m'   # 'LaiStdDev_500m'
scaling_factor = 0.01            # 0.1
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
xarray_data = lai.to_dataset(name="lai")
xarray_data["LaiStdDev_Weighted"] = lai_stddev_weighted

# Save the updated dataset to a Cloud Optimized GeoTIFF
fname = lai_file.replace(".tif", "_qa_weighted_std_dev.tif")
print(fname)
xarray_data["LaiStdDev_Weighted"].rio.to_raster(
    fname, 
    driver="COG", 
    compress="DEFLATE"
)

# Print confirmation
print("Saved the weighted LAI stddev as a Cloud Optimized GeoTIFF.")