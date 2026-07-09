import os
import requests
from tqdm import tqdm

# 1. Define the URLs for the Inside Airbnb Amsterdam release
BASE_URL = "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2026-06-15"

FILES_TO_DOWNLOAD = {
    "listings.csv.gz": f"{BASE_URL}/data/listings.csv.gz",
    "calendar.csv.gz": f"{BASE_URL}/data/calendar.csv.gz",
    "reviews.csv": f"{BASE_URL}/visualisations/reviews.csv", # Summary file
    "neighbourhoods.csv": f"{BASE_URL}/visualisations/neighbourhoods.csv",
    "neighbourhoods.geojson": f"{BASE_URL}/visualisations/neighbourhoods.geojson"
}

def download_file(url, destination_path):
    """Downloads a file in chunks with a visual progress bar."""
    
    # Check if file already exists so we don't redownload it
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
        print(f"Skipping download, file already exists: {destination_path}")
        return

    print(f"Starting download from: {url}")
    
    # stream=True opens the connection but doesn't download the file body yet
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raises an error if download fails (e.g., 404)
    
    # Get total file size from HTTP header (defaults to 0 if not provided)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 * 1024  # 1 MegaByte chunks
    
    # Open the destination file in Write-Binary ('wb') mode
    with open(destination_path, "wb") as file, tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True, 
        desc=os.path.basename(destination_path)
    ) as progress_bar:
        # Loop through chunks and write to disk
        for chunk in response.iter_content(chunk_size):
            written_bytes = file.write(chunk)
            progress_bar.update(written_bytes)

def main():
    # Define directory to save raw data
    raw_data_dir = "data/raw"
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Download each file
    for filename, url in FILES_TO_DOWNLOAD.items():
        destination = os.path.join(raw_data_dir, filename)
        try:
            download_file(url, destination)
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")

if __name__ == "__main__":
    main()