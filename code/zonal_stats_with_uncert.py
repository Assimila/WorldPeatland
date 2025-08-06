
import os
import sys
import pystac
import rioxarray
import xarray as xr
from glob import glob
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CATALOG_URL = "https://s3.waw3-2.cloudferro.com/swift/v1/wpl-stac/stac/catalog.json"

def read_stac_data(site, variable):
    """
    Reads ...
    """
    root: pystac.Catalog = pystac.read_file(CATALOG_URL) 

    # Get the sub-catalog for the site
    catalog: pystac.Catalog = root.get_child(site)

    # Get the collection for the corresponding variable
    collection: pystac.Collection = catalog.get_child(variable)

    # This dataset is chunked for spatial reads
    asset = collection.assets[f"{variable}.xy.zarr"]
    
    ds = xr.open_dataset(
        asset.href,
        **asset.ext.xarray.open_kwargs,  # type: ignore
    )

    return ds

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

def get_pixel_indices_within_classification(dataarray, classification_path):
    """
    Returns the indices (band, y, x) of pixels in dataarray that overlap with 
    the classification in classification_path.
    """
    # Read the classification raster
    classification = rioxarray.open_rasterio(classification_path)

    # Find indices where classification is not nan (i.e., inside the classification)
    indices = np.argwhere(classification[0].values==1)

    return indices

def get_pixel_indices_within_geometry(dataarray, shapefile_path):
    """
    Returns the indices (band, y, x) of pixels in dataarray that overlap with 
    the geometry in shapefile_path.
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Reproject geometry to match raster CRS
    gdf = gdf.to_crs(dataarray.rio.crs)

    # Rasterize the geometry to the shape of the dataarray
    mask = dataarray.rio.clip(gdf.geometry.apply(mapping), gdf.crs,
            all_touched=True, drop=False, invert=False)

    # Find indices where mask is not nan (i.e., inside the geometry)
    indices = np.argwhere(~np.isnan(mask.values))

    return indices

def get_weighted_mean_and_variance(data, indices, spatial_ratio=25):
    """
    Calculate weighted mean and variance for the data at the specified indices using uncertainty as weights.
    Weights are computed as the inverse of the square of uncertainty values.
    Also computes an uncertainty ratio based on weight distribution.
    Returns a DataFrame with the weighted mean, and uncertainty for each time step.
    """ 
    if len(indices) == 0:
        return None, None

    _variable, _uncertainty = list(data.keys())

    weighted_means = []
    weighted_variances = []
    uncertainties = []

    for i in range(data[_variable].shape[0]):
        print(f"Processing time step {i+1}/{data[_variable].shape[0]}...")

        # Select pixels where classification is 1
        data_vals = data[_variable].values[i, indices[:, 0], indices[:, 1]]
        unc_vals = data[_uncertainty].values[i, indices[:, 0], indices[:, 1]]
        
        # Compute weights
        weights = 1.0 / (unc_vals ** 2)

        # Avoid division by zero or nan
        # valid = np.isfinite(data_vals) & np.isfinite(weights) & (weights > 0)
        valid = np.isfinite(data_vals) & np.isfinite(weights) & (weights > 0)
        data_vals = data_vals[valid]
        weights = weights[valid]

        if weights.size == 0:
            weighted_mean = np.nan
            uncertainty = np.nan
        else:
            # Calculate weighted mean
            weighted_mean = np.sum(data_vals * weights) / np.sum(weights)

            # Calculate uncertainty
            unique_weights, counts = np.unique(weights, return_counts=True)
           
            # Numerator: for each unique weight, spatial_ratio^2 / count occurrences, multiply by weight, sum
            numerator = np.sum((spatial_ratio**2 / counts) * unique_weights)

            # Denominator: for each unique weight, count occurrences, multiply by weight, sum
            denominator = np.sum(counts * unique_weights)
            
            # Compute the ratio
            uncertainty = numerator / denominator if denominator != 0 else np.nan

            print(f"Weighted mean: {weighted_mean}, Uncertainty: {uncertainty}")

        weighted_means.append(weighted_mean)
        uncertainties.append(uncertainty)

    # Create a DataFrame to stores the weighted means, variances, and uncertainty ratios
    df = pd.DataFrame({"weighted_mean": weighted_means,
                       "uncertainty": uncertainties},
                      index=data['time'].values)
    
    return df

def create_plot(weighted_mean, weighted_variance, variable):
    """
    Create a plot of the zonal stats time series
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot weighted mean time series
    ax.plot(weighted_mean.index,
            weighted_mean,
            label=f"{variable} - weighted mean", color="C0")

    # Compute standard deviation from variance
    std = np.sqrt(weighted_variance.values)

    # Fill between mean ± std
    ax.fill_between(
        weighted_mean.index,
        weighted_mean - std,
        weighted_mean + std,
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

def extract_zonal_stats(variable, site,
                        classification_fname, plot=False):
    """
    Extract zonal stats using associated uncertainties
        The stats then will be linearly interpolated to create
        synthetic daily data
    """
    # Read data and associated uncertainty
    data = read_stac_data(variable=variable, site=site)

    # Get the indices within the geometry
    indices = get_pixel_indices_within_classification(data, classification_fname)

    # Compute stats
    weighted_stats = get_weighted_mean_and_variance(data, indices)

    # Linear interpolation to create synthetic daily data
    weighted_mean = weighted_stats['weighted_mean'].resample('D').interpolate('linear')
    uncertainty = weighted_stats['uncertainty'].resample('D').interpolate('linear')

    if plot == True:
        create_plot(weighted_mean, uncertainty, variable)

    return weighted_mean, uncertainty


if __name__ == "__main__":

    if len(sys.argv) != 3:

        # Check inputs
        print((f"Usage: python .zonal_stats_with_uncert.py"
               f"<site_root_data_dir> <peatland_extent_path>"))
    else:
        # Site name e.g. Degero
        # site = sys.argv[1]
        site = "degero"

        # Full path of the peatland classification GeoTiff
        # classification_fname = sys.argv[2]
        classification_fname = "/wp_data/sites/Degero/WhatSARpeat/WhatSARPeat2024_Degero.tif"

        variables = ['lai', 'fpar', 'albedo',
                    'evi', 'lst-day',
                    'lst-night', 'lst-diurnal-range']

        data = pd.DataFrame()
        uncertainty = pd.DataFrame()

        for variable in variables:
            print(f"Processing {variable}...")
            w_mu, w_unc = extract_zonal_stats(variable, site,
                                              classification_fname, plot=True)
            print(w_mu, w_unc)

            data[variable] = w_mu
            uncertainty[variable] = w_unc

        filename = "time_series.h5"
        data.to_hdf(filename, key="data")
        uncertainty.to_hdf(filename, key="uncertainty")
