
import os
import sys
import rioxarray
from glob import glob
from osgeo import gdal
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import mapping
import matplotlib.pyplot as plt

def read_data_and_uncertainty(data_path, uncertainty_path):
    """
    Reads WorldPeatland data and associated uncertainty GeoTIFFs as xarray.DataArray.
    """
    lai = rioxarray.open_rasterio(data_path)
    uncertainty = rioxarray.open_rasterio(uncertainty_path)

    # Read per-band metadata for time
    d = gdal.Open(data_path)
 
    bands = d.RasterCount
    band_times = []

    for i in range(1, bands + 1):
        tags = d.GetRasterBand(i).GetMetadata()
        # Try several possible keys for date
        date_str = tags.get("RANGEBEGINNINGDATE") or tags.get("DATE") or tags.get("time") or None
        if date_str is not None:
            # Accept both YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS formats
            date_str = date_str.split("T")[0]
            band_times.append(np.datetime64(date_str))
        else:
            band_times.append(np.datetime64('NaT'))

    # Rename dimensions and assign coordinates
    lai = lai.rename({"band": "time", "y": "latitude", "x": "longitude"})
    uncertainty = uncertainty.rename({"band": "time", "y": "latitude", "x": "longitude"})
    lai = lai.assign_coords(time=("time", band_times))
    uncertainty = uncertainty.assign_coords(time=("time", band_times))

    return lai, uncertainty

def get_pixel_indices_within_geometry(dataarray, shapefile_path):
    """
    Returns the indices (band, y, x) of pixels in dataarray that overlap with the geometry in shapefile_path.
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Reproject geometry to match raster CRS
    gdf = gdf.to_crs(dataarray.rio.crs)
    # Rasterize the geometry to the shape of the dataarray
    mask = dataarray.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=False, invert=False)
    # Find indices where mask is not nan (i.e., inside the geometry)
    indices = np.argwhere(~np.isnan(mask.values))
    return indices

def get_weighted_mean(data, uncertainty, indices):
    """
    Calculate weighted mean for the data at the specified indices using uncertainty as weights.
    Weights are computed as the inverse of the square of uncertainty values.
    Returns a DataFrame with the weighted mean for each time step and the weights applied.
    """ 
    if len(indices) == 0:
        return None, None

    weighted_means = []

    for i in range(data.shape[0]):    
        # Select pixels inside geometry
        data_vals = data.values[i, indices[:, 0], indices[:, 1]]
        unc_vals = uncertainty.values[i, indices[:, 0], indices[:, 1]]

        # Compute weights
        weights = 1.0 / (unc_vals ** 2)
        # Avoid division by zero or nan
        valid = np.isfinite(data_vals) & np.isfinite(weights) & (weights > 0)
        data_vals = data_vals[valid]
        weights = weights[valid]

        if weights.size == 0:
            weighted_mean = np.nan
        else:
            weighted_mean = np.sum(data_vals * weights) / np.sum(weights)

        weighted_means.append(weighted_mean)

    # Create a DataFrame to stores the weighted means
    df = pd.DataFrame({"weighted_mean": weighted_means}, index=data['time'].values)
    
    return df

def get_weighted_variance(data, uncertainty, indices, weighted_mean):
    """
    Calculate weighted variance for the data at the specified indices using uncertainty as weights.
    Weights are computed as the inverse of the square of uncertainty values.
    Returns a DataFrame with the weighted variance for each time step.
    """ 
    if len(indices) == 0:
        return None, None

    weighted_variances = []

    for i in range(data.shape[0]):    
        # Select pixels inside geometry
        data_vals = data.values[i, indices[:, 0], indices[:, 1]]
        unc_vals = uncertainty.values[i, indices[:, 0], indices[:, 1]]
        
        # Compute weights
        weights = 1.0 / (unc_vals ** 2)
        # Avoid division by zero or nan
        valid = np.isfinite(data_vals) & np.isfinite(weights) & (weights > 0)
        data_vals = data_vals[valid]
        weights = weights[valid]

        if weights.size == 0:
            weighted_mean = np.nan
        else:
            weighted_spread = np.sum((weights * (data_vals - weighted_mean['weighted_mean'][i])) ** 2)
            weights_modulator = np.sum(weights)

            weighted_variance = weighted_spread * (1.0 / (weights_modulator ** 2))

        weighted_variances.append(weighted_variance)

    # Create a DataFrame to stores the weighted means
    df = pd.DataFrame({"weighted_variance": weighted_variances}, index=data['time'].values)
    
    return df


def create_plot(weighted_mean, weighted_variance, variable):
    """
    Create a plot of the zonal stats time series
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot weighted mean time series
    ax.plot(weighted_mean.index,
            weighted_mean['weighted_mean'],
            label=f"{variable} - weighted mean", color="C0")

    # Compute standard deviation from variance
    std = np.sqrt(weighted_variance['weighted_variance'].values)

    # Fill between mean ± std
    ax.fill_between(
        weighted_mean.index,
        weighted_mean['weighted_mean'] - std,
        weighted_mean['weighted_mean'] + std,
        color="C0",
        alpha=0.3,
       label=f"{variable} - weighted std dev"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(variable)
    ax.set_title(f"{variable}")
    ax.legend()
    fig.autofmt_xdate()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"/tmp/{variable}_weighted_mean_and_uncert.png", dpi=150)

def extract_zonal_stats(variable, site_directory,
                        shp_fname, plot=False):
    """
    Extract zonal stats using associated uncertainties
        The stats then will be linearly interpolated to create
        synthetic daily data
    """
    # Data
    fname = f'*._{variable}.linear.smoothn.*.descaled.tif'
    data_fname =  glob(os.path.join(site_directory, fname))[0]
    if len(data_fname) == 0:
        print(f"File {data_fname} not found")
        return

    # Uncertainty
    fname = f'*._{variable}.linear.smoothn.*_qa_weighted_std_dev.tif'
    uncertainty_fname = glob(os.path.join(site_directory, fname))[0]
    if len(uncertainty_fname) == 0:
        print(f"File {uncertainty_fname} not found")
        return

    # Read data and associated uncertainty
    data, uncertainty = read_data_and_uncertainty(data_fname,
                                                  uncertainty_fname)
    # Get the indices within the geometry
    indices = get_pixel_indices_within_geometry(data[0], shp_fname)

    # Compute stats
    weighted_mean = get_weighted_mean(data, uncertainty, indices)
    weighted_variance = get_weighted_variance(data, uncertainty,
                                              indices, weighted_mean)

    # Linear interpolation to create synthetic daily data
    weighted_mean = weighted_mean.resample('D').interpolate('linear')
    weighted_variance = weighted_variance.resample('D').interpolate('linear')

    if plot == True:
        create_plot(weighted_mean, weighted_variance, variable)

    return weighted_mean, weighted_variance


if __name__ == "__main__":

    if len(sys.argv) != 3:

        # Check inputs
        print((f"Usage: python .zonal_stats_with_uncert.py"
               f"<site_root_data_dir> <shapefile_path>"))
    else:
        site_directory = sys.argv[1]
        shapefile_path = sys.argv[2]

        variables = ['Lai_500m', 'Fpar_500m', 'Albedo_WSA_Band2',
                    '1_km_16_days_EVI', 'LST_Day_1km',
                    'LST_Night_1km', 'LST_Diurnal_1km']

        variables = ['Lai_500m', 'Fpar_500m', 'Albedo_WSA_Band2',
                    '1_km_16_days_EVI', 'LST_Day_1km',
                    'LST_Night_1km']
        
        data = pd.DataFrame()
        variance = pd.DataFrame()

        for variable in variables:
            print(f"Processing {variable}...")
            w_mu, w_var = extract_zonal_stats(variable, site_directory,
                                              shapefile_path, plot=False)
            
            data[variable] = w_mu['weighted_mean']
            variance[variable] = w_var['weighted_variance']

        filename = "time_series.h5"
        data.to_hdf(filename, key="data")
        variance.to_hdf(filename, key="variance")
