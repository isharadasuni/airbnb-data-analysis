import os
import duckdb

def build_database(db_path="data/processed/airbnb_analytics.db", processed_dir="data/processed"):
    print(f"🗄️ Creating SQL Database at {db_path}...")
    
    # Connect to DuckDB database file (creates it if it doesn't exist)
    con = duckdb.connect(db_path)
    
    # Convert file paths to use forward slashes for DuckDB compatibility
    listings_parquet = os.path.join(processed_dir, "listings.parquet").replace('\\', '/')
    calendar_parquet = os.path.join(processed_dir, "calendar.parquet").replace('\\', '/')
    neighbourhoods_parquet = os.path.join(processed_dir, "neighbourhoods.parquet").replace('\\', '/')
    
    # --- STEP 1: CREATE DIMENSION TABLES ---
    
    print("Creating dim_hosts dimension...")
    con.execute("DROP TABLE IF EXISTS dim_hosts;")
    con.execute(f"""
        CREATE TABLE dim_hosts AS
        SELECT DISTINCT
            host_id,
            host_name,
            host_since,
            host_location,
            host_response_time,
            host_response_rate,
            host_acceptance_rate,
            host_is_superhost,
            host_listings_count AS host_total_listings_count
        FROM read_parquet('{listings_parquet}')
        WHERE host_id IS NOT NULL;
    """)
    
    print("Creating dim_listings dimension...")
    con.execute("DROP TABLE IF EXISTS dim_listings;")
    con.execute(f"""
        CREATE TABLE dim_listings AS
        SELECT
            id AS listing_id,
            name,
            host_id,
            neighbourhood_cleansed AS neighbourhood,
            room_type,
            property_type,
            accommodates,
            bedrooms,
            beds,
            price AS listing_price,
            latitude,
            longitude,
            amenities
        FROM read_parquet('{listings_parquet}');
    """)
    
    # --- STEP 2: CREATE FACT TABLES ---
    
    print("Creating fact_calendar fact table...")
    con.execute("DROP TABLE IF EXISTS fact_calendar;")
    # We join with dim_listings to pull listing_price as price
    con.execute(f"""
        CREATE TABLE fact_calendar AS
        SELECT
            c.listing_id,
            c.date,
            c.available,
            l.listing_price AS price,
            CASE WHEN c.available = 0 THEN 1 ELSE 0 END AS occupied
        FROM read_parquet('{calendar_parquet}') c
        JOIN dim_listings l ON c.listing_id = l.listing_id;
    """)
    
    # --- STEP 3: ANALYTICAL DATA ENRICHMENT ---
    print("Enriching listings with calculated occupancy metrics and revenue...")
    
    # Create a temporary table to calculate average metrics per listing
    con.execute("""
        CREATE OR REPLACE TABLE listing_aggregates AS
        SELECT
            listing_id,
            COUNT(*) AS total_days,
            SUM(occupied) AS occupied_days,
            ROUND(SUM(occupied)::DOUBLE / COUNT(*), 4) AS occupancy_rate,
            ROUND(SUM(CASE WHEN occupied = 1 THEN price ELSE 0 END), 2) AS estimated_revenue
        FROM fact_calendar
        GROUP BY listing_id;
    """)
    
    # Add new columns to dim_listings
    con.execute("ALTER TABLE dim_listings ADD COLUMN occupancy_rate DOUBLE;")
    con.execute("ALTER TABLE dim_listings ADD COLUMN estimated_revenue DOUBLE;")
    
    # Update dim_listings table with calculated aggregates
    con.execute("""
        UPDATE dim_listings
        SET 
            occupancy_rate = COALESCE(agg.occupancy_rate, 0.0),
            estimated_revenue = COALESCE(agg.estimated_revenue, 0.0)
        FROM listing_aggregates agg
        WHERE dim_listings.listing_id = agg.listing_id;
    """)
    
    # Drop temporary table
    con.execute("DROP TABLE IF EXISTS listing_aggregates;")
    
    # --- STEP 4: VERIFY ROW COUNTS ---
    print("\n📊 Database Summary:")
    for table_name in ["dim_hosts", "dim_listings", "fact_calendar"]:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
        print(f"• Table '{table_name}': {row_count:,} rows")
        
    con.close()
    print("\n🎉 Database build successfully finished!")

if __name__ == "__main__":
    build_database()