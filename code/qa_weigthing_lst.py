
from glob import glob
import xarray as xr
import rioxarray

site = 'Degero'

# LST
# 'MYD11A1.061', 'LST_Night_1km', 'LST_Night_StdDev_1km', 0.02, 0.5

# Albedo
# 'MCD43A3.061', 'Albedo_WSA_Band2', 'Albedo_WSA_Band2_StdDev_500m', 0.05, 0.001

product = 'MOD13A2.061'
variable = '1_km_16_days_EVI'
uncert_var = '1_km_16_days_EVI_StdDev_1km'
smmoth_factor = 0.5
scaling_factor = 0.0001

weighting_factor = 0.05  # Define the weighting/additive factor
factor_type = 'multiplicative'  # Define the factor type, either 'additive' or 'multiplicative'

# Strahler et al. (1999) and Schaaf et al. (2002) report:
# NIR band white-sky albedo RMSE ~0.02–0.05 in vegetated areas
# nominal_uncert_data = 0.03

nominal_uncert_data = 0.02

# Paths to the VRT file and mask file
data_file = f"/wp_data/sites/{site}/MODIS/{product}/h??v??/{variable}/interpolated/{product}._{variable}.linear.smoothn.{smmoth_factor}.tif"
data_file = glob(data_file)[0] 

mask_file = f"/wp_data/sites/{site}/MODIS/analytics_QA_settings/_{variable}_qa_analytics_mask.tif"
mask_file = glob(mask_file)[0]

# Open the VRT file using rioxarray
data = rioxarray.open_rasterio(data_file, mask_and_scale=True).astype("float32") * scaling_factor

# Deep copy the data to a new variable
data_stddev = data.copy(deep=True)

data_stddev.data[:,:,:] = nominal_uncert_data

# Open the mask file using rioxarray
mask = rioxarray.open_rasterio(mask_file)

# Check if both layers have the same shape
if data.shape == mask.shape:
    print("The data and mask have the same shape.")
else:
    print("The data and mask do not have the same shape.")
    print(f"Data shape: {data.shape}")
    print(f"Mask shape: {mask.shape}")
    raise ValueError("Data and mask shapes do not match.")

# Apply the weighting factor where the mask is 0, keep original values where mask is 1
if factor_type == 'additive':
    data_stddev_weighted = xr.where(mask == 0, data + weighting_factor, data_stddev)
elif factor_type == 'multiplicative':
    data_stddev_weighted = xr.where(mask == 0, data * weighting_factor, data_stddev)

# Add the new weighted LAI stddev as a new variable in the dataset
xarray_data = data.to_dataset(name="data")
xarray_data[uncert_var] = data_stddev_weighted

# Save the updated dataset to a Cloud Optimized GeoTIFF
fname = data_file.replace(".tif", "_qa_weighted_std_dev.tif")
print(fname)
xarray_data[uncert_var].rio.to_raster(
    fname, 
    driver="COG", 
    compress="DEFLATE"
)

# Print confirmation
print("Saved the weighted stddev as a Cloud Optimized GeoTIFF.")