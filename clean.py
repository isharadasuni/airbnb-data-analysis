import os
import pandas as pd
import numpy as np

def clean_price(price_str):
    """Converts a currency string like '$1,200.00' to a float."""
    if pd.isna(price_str):
        return np.nan
    # Remove currency symbol ($), commas, and spaces
    clean_str = str(price_str).replace('$', '').replace(',', '').strip()
    try:
        return float(clean_str)
    except ValueError:
        return np.nan

def clean_listings(raw_path, processed_path):
    """Cleans listings data and saves to parquet format."""
    print("🧹 Cleaning listings dataset...")
    df = pd.read_csv(raw_path, low_memory=False)
    
    # Clean price column
    df['price'] = df['price'].apply(clean_price)
    
    # Parse dates
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    
    # Standardize rates (convert '95%' -> 0.95)
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
            
    # Impute missing values with medians
    df['beds'] = df['beds'].fillna(df['beds'].median())
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median() if not pd.isna(df['bedrooms'].median()) else 1.0)
    
    # Filter valid coordinates (Amsterdam bounding box)
    lat_valid = df['latitude'].between(52.0, 53.0)
    lon_valid = df['longitude'].between(4.5, 5.5)
    df = df[lat_valid & lon_valid]
    
    # Remove negative or free pricing records
    df = df[df['price'] > 0]
    
    # Save cleaned listings as Parquet (optimized column format)
    df.to_parquet(processed_path, index=False)
    print(f"✅ Cleaned listings saved. Shape: {df.shape}")

def clean_calendar(raw_path, processed_path):
    """Cleans calendar availability data."""
    print("🧹 Cleaning calendar dataset...")
    
    chunks = []
    # Read calendar in chunks of 1M rows to avoid memory crashes
    for chunk in pd.read_csv(raw_path, chunksize=1000000, low_memory=False):
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        # available: 't'/'f' -> 1/0
        chunk['available'] = chunk['available'].map({'t': 1, 'f': 0}).fillna(0).astype(np.int8)
        chunk['listing_id'] = chunk['listing_id'].astype(np.int64)
        chunks.append(chunk)
        
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(processed_path, index=False)
    print(f"✅ Cleaned calendar saved. Shape: {df.shape}")

def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Run cleaners
    clean_listings(os.path.join(raw_dir, "listings.csv.gz"), os.path.join(processed_dir, "listings.parquet"))
    clean_calendar(os.path.join(raw_dir, "calendar.csv.gz"), os.path.join(processed_dir, "calendar.parquet"))
    
    # Clean simple neighbourhoods files
    n_df = pd.read_csv(os.path.join(raw_dir, "neighbourhoods.csv"))
    n_df.to_parquet(os.path.join(processed_dir, "neighbourhoods.parquet"), index=False)
    
    print("🎉 Stage 2 Cleaning Complete!")

if __name__ == "__main__":
    main()