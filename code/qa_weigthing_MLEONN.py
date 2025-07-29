
import os.path as osp
import sys
from glob import glob
import xarray as xr
import rioxarray
import rasterio
import osgeo.gdal as gdal

site = sys.argv[1]

# MODIS LST, nominal_uncert_data ~2.0K
lst_day = {'product' : 'MYD11A1.061',
           'variable' : 'LST_Day_1km',
           'uncert_var' : 'LST_Day_StdDev_1km',
           'smooth_factor' : 0.5,
           'scaling_factor' : 0.02,
           'weighting_factor' : 0.02,
           'nominal_uncert_data' : 2.0}

lst_night = {'product' : 'MYD11A1.061',
             'variable' : 'LST_Night_1km',
             'uncert_var' : 'LST_Night_StdDev_1km',
             'smooth_factor' : 0.5,
             'scaling_factor' : 0.02,
             'weighting_factor' : 0.02,
             'nominal_uncert_data' : 2.0}

# Strahler et al. (1999) and Schaaf et al. (2002) report:
# NIR band white-sky albedo RMSE ~0.02–0.05 in vegetated areas
# nominal_uncert_data = 0.03
albedo = {'product' : 'MCD43A3.061',
          'variable' : 'Albedo_WSA_Band2',
          'uncert_var' : 'Albedo_WSA_Band2_StdDev_500m',
          'smooth_factor' : 0.5,
          'scaling_factor' : 0.001,
          'weighting_factor' : 0.02,
          'nominal_uncert_data' : 0.03}

evi = {'product' : 'MOD13A2.061',
       'variable' : '1_km_16_days_EVI',
       'uncert_var' : '1_km_16_days_EVI_StdDev_1km',
       'smooth_factor' : 0.5,
       'scaling_factor' : 0.0001,
       'weighting_factor' : 0.02,
       'nominal_uncert_data' : 0.05}

products = [lst_day, lst_night, albedo, evi]

for _product in products:

    product = _product['product']
    variable = _product['variable']
    uncert_var = _product['uncert_var']
    smooth_factor = _product['smooth_factor']
    scaling_factor = _product['scaling_factor']
    weighting_factor = _product['weighting_factor']
    nominal_uncert_data = _product['nominal_uncert_data']

    # Paths to the VRT file and mask file
    data_file = f"/wp_data/sites/{site}/MODIS/{product}/h??v??/{variable}/interpolated/{product}._{variable}.linear.smoothn.{smooth_factor}.tif"
    data_file = glob(data_file)[0] 

    mask_file = f"/wp_data/sites/{site}/MODIS/analytics_QA_settings/_{variable}_qa_analytics_mask.tif"
    mask_file = glob(mask_file)[0]

    output_dir = f'/wp_data/sites/{site}/MODIS/timeSeries'
    output_fname = osp.basename(data_file.replace(".tif", "_qa_weighted_std_dev.tif"))
    output_fname = osp.join(output_dir, output_fname)

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

    # Set uncertainties
    data_stddev_weighted = xr.where(mask == 0, data * weighting_factor, data_stddev)

    # Add the new weighted variable stddev as a new variable in the dataset
    xarray_data = data.to_dataset(name="data")
    xarray_data[uncert_var] = data_stddev_weighted

    # Extract per-band tags using rasterio
    with rasterio.open(data_file) as src:
        tags_per_band = [src.tags(i) for i in range(1, src.count + 1)]
        geotransform = src.transform.to_gdal()
        projection = src.crs.to_wkt()

    bands, height, width = xarray_data[uncert_var].data.shape

    # Save the updated dataset to a Cloud Optimized GeoTIFF
    print(output_fname)

    # Create output with GDAL
    driver = gdal.GetDriverByName("GTiff")
    driver_options = ['COMPRESS=DEFLATE',
                      'BIGTIFF=YES',
                      'PREDICTOR=1',
                      'TILED=YES',
                      'COPY_SRC_OVERVIEWS=YES']

    out_ds = driver.Create(output_fname, width, height, bands, gdal.GDT_Float32) #,
    #    options=driver_options)

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

    # Print confirmation
    print("Saved the weighted stddev as a Cloud Optimized GeoTIFF.")
