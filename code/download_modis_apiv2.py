
"""
MODIS MCD15A3H Downloader using LAADS Web API v2
This script downloads MCD15A3H (LAI/FPAR) data using the new API v2 endpoints.
"""

import requests
import os
import json
import sys
from urllib.parse import urlencode, urljoin
from datetime import datetime, timedelta
import time
import calendar

def read_app_key(token_file="token"):
    """
    Read the LAADS Web APP-KEY from a local file.
    
    Args:
        token_file (str): Path to the file containing the APP-KEY
        
    Returns:
        str: The APP-KEY
        
    Raises:
        FileNotFoundError: If the token file doesn't exist
        ValueError: If the token file is empty or contains invalid content
    """
    try:
        with open(token_file, 'r') as f:
            app_key = f.read().strip()

        if not app_key:
            raise ValueError(f"Token file '{token_file}' is empty")

        # Basic validation - LAADS keys are typically long alphanumeric strings
        if len(app_key) < 10:
            raise ValueError(f"Token in '{token_file}' appears to be too short")

        print(f"Successfully read APP-KEY from '{token_file}'")
        return app_key

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Token file '{token_file}' not found. "
            f"Please create this file with your LAADS Web APP-KEY.\n"
            f"Get your APP-KEY from: https://ladsweb.modaps.eosdis.nasa.gov/profile/"
        )
    except Exception as e:
        raise ValueError(f"Error reading token file '{token_file}': {e}")

class LADSWebAPIv2:
    """LAADS Web API v2 client for downloading MODIS data."""

    def __init__(self, app_key):
        self.app_key = app_key
        self.base_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {app_key}',
            'User-Agent': 'MODIS-Downloader-APIv2/1.0'
        })

    def search_files(self, product, collection, start_date, end_date, tile=None, bbox=None):
        """
        Search for files using the API v2 content/details endpoint.
        
        Args:
            product (str): Product name (e.g., 'MCD15A3H')
            collection (str): Collection version (e.g., '6', '61')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            tile (str, optional): MODIS tile (e.g., 'h18v04')
            bbox (dict, optional): Bounding box {'west': lon, 'east': lon, 'north': lat, 'south': lat}
        
        Returns:
            list: List of file details
        """
        # Construct query parameters
        params = {
            'products': product,
            'temporalRanges': f'{start_date}..{end_date}'
        }

        # Add spatial constraint
        if tile:
            # For tile-based filtering, we'll need to filter results after retrieval
            # as the API doesn't directly support tile filtering in v2
            pass
        elif bbox:
            # Construct bounding box string for API v2
            bbox_str = f"[BBOX]W{bbox['west']} N{bbox['north']} E{bbox['east']} S{bbox['south']}"
            params['regions'] = bbox_str

        # Make request to content/details endpoint
        url = f"{self.base_url}/content/details"

        try:
            print(f"Searching for {product} files from {start_date} to {end_date}...")
            response = self.session.get(url, params=params)
            response.raise_for_status()

            files = response.json()

            # Filter by tile if specified (since API v2 doesn't support direct tile filtering)
            filtered_files = []

            for file_info in files['content']:
                filename = file_info.get('name', '')
                start = file_info.get('start', '').split(' ')[0]  # extract date portion
                if tile in filename and start_date == start:
                    filtered_files.append(file_info)

            files = filtered_files

            print(f"Found {len(files)} files matching criteria")
            return files

        except requests.exceptions.RequestException as e:
            print(f"Error searching for files: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing API response: {e}")
            return []

    def get_download_url(self, file_path):
        """
        Get download URL for a file using the API v2 archives endpoint.
        
        Args:
            file_path (str): Full path to the file in the archive
            
        Returns:
            str: Download URL
        """
        # For API v2, we can directly stream files from the archives endpoint
        return f"{self.base_url}/content/archives/{file_path}"

    def download_file(self, file_info, output_dir):
        """
        Download a single file.
        
        Args:
            file_info (dict): File information from search results
            output_dir (str): Directory to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filename = file_info.get('name')
            file_path = file_info.get('path', file_info.get('fileId', ''))

            if not filename:
                print(f"No filename found in file info: {file_info}")
                return False

            # Get download URL
            download_url = self.get_download_url(file_path)

            # Full path for saved file
            filepath = os.path.join(output_dir, filename)

            print(f"Downloading {filename}...")

            # Download with streaming to handle large files
            response = self.session.get(download_url, stream=True)
            response.raise_for_status()

            # Get file size if available
            file_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress indication
                        if file_size > 0:
                            progress = (downloaded / file_size) * 100
                            print(f"\r  Progress: {progress:.1f}%", end='', flush=True)

            print(f"\n✓ Downloaded: {filename}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"\n✗ Failed to download {filename}: {e}")
            return False
        except Exception as e:
            print(f"\n✗ Error downloading {filename}: {e}")
            return False

def download_modis_data(product="MCD15A3H", collection="6", tile="h18v04",
                       years=None, tile_dir="./MCD15A3H_data"):
    """
    Main function to download MODIS data using API v2.
    
    Args:
        product (str): MODIS product name
        collection (str): Collection version
        tile (str): MODIS tile
        years (list): List of years to download
        output_dir (str): Output directory
    """
    if years is None:
        years = [2023]  # Default to current year

    # Read APP-KEY from token file
    try:
        app_key = read_app_key()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return False

    # Create API client
    client = LADSWebAPIv2(app_key)

    # Create output directory
    os.makedirs(tile_dir, exist_ok=True)

    total_downloaded = 0
    total_files = 0

    for year in years:
        for month in range(1,12):
            for day in range(1, calendar.monthrange(year, month)[1]+1):
                print(f"\n{'='*50}")
                print(f"Processing year {year}")
                print(f"{'='*50}")

                # Define date range for the year
                start_date = f"{year}-{month:02}-{day:02}"
                # end_date = f"{year}-12-31"
                end_date = f"{year}-{month:02}-{day:02}"

                # Search for files
                files = client.search_files(
                    product=product,
                    collection=collection,
                    start_date=start_date,
                    end_date=end_date,
                    tile=tile
               )

                if not files:
                    print(f"No files found for {year}")
                    continue

                # Download files
                year_downloaded = 0
                for i, file_info in enumerate(files, 1):
                    print(f"\n[{i}/{len(files)}] ", end='')

                    if client.download_file(file_info, tile_dir):
                        year_downloaded += 1
                        total_downloaded += 1

                    # Small delay to be respectful to the server
                    time.sleep(2)

        total_files += len(files)
        print(f"\nYear {year} summary: {year_downloaded}/{len(files)} files downloaded")

    # Final summary
    print(f"\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"Total files found: {total_files}")
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Success rate: {(total_downloaded/total_files*100) if total_files > 0 else 0:.1f}%")
    print(f"Files saved to: {tile_dir}")

    return total_downloaded > 0


def download_modis_data_interval(product="MCD15A3H", collection="6", tile="h18v04",
                        start_date=None, end_date=None, tile_dir="./MCD15A3H_data"):
    """
    Main function to download MODIS data using API v2.

    Args:
        product (str): MODIS product name
        collection (str): Collection version
        tile (str): MODIS tile
        years (list): List of years to download
        output_dir (str): Output directory
    """
    # Read APP-KEY from token file
    try:
        app_key = read_app_key()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return False

    # Create API client
    client = LADSWebAPIv2(app_key)

    # Create output directory
    os.makedirs(tile_dir, exist_ok=True)

    total_downloaded = 0
    total_files = 0

    print(f"{'=' * 50}")

    # Search for files
    files = client.search_files(
        product=product,
        collection=collection,
        start_date=start_date,
        end_date=end_date,
        tile=tile
    )

    # Download files
    year_downloaded = 0
    for i, file_info in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] ", end='')

        if client.download_file(file_info, tile_dir):
            year_downloaded += 1
            total_downloaded += 1

        # Small delay to be respectful to the server
        time.sleep(0.5)

        total_files += len(files)

    # Final summary
    print(f"\n{'=' * 50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total files found: {total_files}")
    print(f"Total files downloaded: {total_downloaded}")
    print(f"Success rate: {(total_downloaded / total_files * 100) if total_files > 0 else 0:.1f}%")
    print(f"Files saved to: {tile_dir}")

    return total_downloaded > 0


def create_token_file_example():
    """Create an example token file if it doesn't exist."""
    token_file = "token"
    if not os.path.exists(token_file):
        with open(token_file, 'w') as f:
            f.write("# LAADS Web APP-KEY for API v2\n")
            f.write("# Get your APP-KEY from: https://ladsweb.modaps.eosdis.nasa.gov/profile/\n")
            f.write("# Replace the line below with your actual APP-KEY\n")
            f.write("YOUR_APP_KEY_HERE\n")
        print(f"Created example token file: {token_file}")
        print("Please edit this file and replace 'YOUR_APP_KEY_HERE' with your actual APP-KEY")
        return False
    return True

if __name__ == "__main__":

    print("MODIS MCD15A3H Downloader - API v2")
    print("=" * 50)

    CONFIG = {
        "product": "MCD15A3H",
        "collection": "61",
        "tile": "h18v04",
        "years": [2024, 2025],
        "tile_dir": "./MCD15A3H_data"
    }


    # Check if token file exists, create example if not
    if not create_token_file_example():
        exit(1)

    # Display configuration
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Download data
    success = download_modis_data(**CONFIG)

    if success:
        print("\n✓ Download process completed successfully!")
    else:
        print("\n✗ Download process failed. Please check your configuration and try again.")
        exit(1)
