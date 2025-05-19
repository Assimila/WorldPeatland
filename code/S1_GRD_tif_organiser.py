import os
import re
from collections import defaultdict
from osgeo import gdal

"""
**Script Description:**

This script is designed to reorganize and merge Sentinel-1 S1\_GRD monthly `.tif` files that have been downloaded from 
Google Earth Engine (GEE) in multiple chunks due to large areas of interest (AOIs).

**Pre-processing Step (Manual):**
Before running the script, manually consolidate the `.tif` files into three main directories:

* `VV_ASCENDING`
* `VH_ASCENDING`
* `angle_ASCENDING`
  (Repeat for DESCENDING if applicable: `VV_DESCENDING`, `VH_DESCENDING`, `angle_DESCENDING`)

GEE typically splits large AOIs into smaller tiles, creating separate `VV`, `VH`, and `angle` folders for each region.
 Move all `.tif` files of the same type (e.g., all VV) into their respective single folders.

**Script Functionality:**
The script merges `.tif` files corresponding to different regions but from the same month and year into a single `.tif` 
file, preserving important metadata such as timestamp, coordinate reference system (CRS), and other geospatial 
attributes.

"""

# Parent directory containing the subfolders
parent_folder = "/wp_data/sites/CentralKalimantan/Sentinel/datacube/S1_GRD"

# Regex pattern to extract the year-month from filenames
date_pattern = re.compile(r"_(\d{4}-\d{2})\.")  # Match pattern for year-month in filenames


# Function to extract band metadata from a single file
def extract_band_metadata(tiff_file):
    band_metadata = {}
    dataset = gdal.Open(tiff_file)
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        metadata = band.GetMetadata()
        band_metadata[i] = metadata
    dataset = None  # Close the dataset
    return band_metadata


# Function to set band metadata in a merged TIFF
def set_band_metadata(tiff_file, band_metadata_list):
    """
    @param tiff_file:
    @param band_metadata_list: is a list of dictionary metadata per band
    @return:
    """
    dataset = gdal.Open(tiff_file, gdal.GA_Update)
    for i, metadata in enumerate(band_metadata_list, start=1):
        band = dataset.GetRasterBand(i)
        band.SetMetadata(metadata)
    dataset.FlushCache()
    dataset = None  # Close the dataset


# Iterate through each main subdirectory in the parent folder
for subfolder in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder)

    # Check if it is a directory
    if os.path.isdir(subfolder_path):
        # Iterate through the "home folders" in this subdirectory
        for home_folder in os.listdir(subfolder_path):
            home_folder_path = os.path.join(subfolder_path, home_folder)

            # Check if it's a directory
            if os.path.isdir(home_folder_path):
                print(f"Processing folder: {home_folder_path}")

                # Create a dictionary to group files by year-month
                files_by_date = defaultdict(list)

                # Loop through files in the current home folder
                for file in os.listdir(home_folder_path):
                    if file.endswith(".tif"):
                        match = date_pattern.search(file)
                        if match:
                            year_month = match.group(1)
                            files_by_date[year_month].append(os.path.join(home_folder_path, file))

                # Process each group of files
                for year_month, files in files_by_date.items():
                    if len(files) == 1:
                        # If there's only one file, rename it
                        old_file = files[0]
                        new_file = os.path.join(home_folder_path, f"S1_GRD_{subfolder}_{year_month}.tif")
                        print(f"Renaming: {old_file} -> {new_file}")
                        os.rename(old_file, new_file)
                    else:
                        # If there are multiple files, merge them
                        output_file = os.path.join(home_folder_path, f"S1_GRD_{subfolder}_{year_month}.tif")

                        # Extract metadata only from the first file
                        print(f"Extracting metadata from the first file: {files[0]}")
                        band_metadata = extract_band_metadata(files[0])  # Extract metadata from the first file only

                        # Merge files using gdal_merge.py
                        input_files = " ".join(files)
                        command = f"gdal_merge.py -o {output_file} -of GTiff -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES {input_files}"
                        print(f"Running: {command}")
                        os.system(command)

                        # Apply the extracted band metadata to the merged TIFF
                        print(f"Setting band metadata for: {output_file}")

                        # band_metadata is a dict of dictionaries turn it to a list
                        band_metadata_list = [band_metadata[key] for key in sorted(band_metadata.keys())]
                        set_band_metadata(output_file, band_metadata_list)

# TODO delete the initial files
