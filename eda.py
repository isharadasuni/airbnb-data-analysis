import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional style for charts
sns.set_theme(style="whitegrid")

def generate_charts(db_path="data/processed/airbnb_analytics.db", plot_dir="data/processed/plots"):
    print("📈 Generating Exploratory Data Analysis plots...")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Connect to the DuckDB database
    con = duckdb.connect(db_path)
    
    # 1. Fetch listings data
    df_listings = con.execute("""
        SELECT listing_price AS price, room_type, neighbourhood 
        FROM dim_listings;
    """).df()
    
    # 2. Fetch daily calendar pricing data
    df_calendar = con.execute("""
        SELECT date, AVG(price) AS avg_price 
        FROM fact_calendar 
        GROUP BY date 
        ORDER BY date;
    """).df()
    df_calendar['date'] = pd.to_datetime(df_calendar['date'])
    
    con.close()
    
    # --- CHART 1: Price Distribution (< €800 to avoid extreme outliers) ---
    print("Saving price_distribution.png...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_listings[df_listings['price'] < 800], x='price', kde=True, bins=40, color='#1f77b4')
    plt.title("Distribution of Nightly Airbnb Prices in Amsterdam")
    plt.xlabel("Price (€)")
    plt.ylabel("Count of Listings")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "price_distribution.png"), dpi=300)
    plt.close()
    
    # --- CHART 2: Boxplot of Price by Room Type ---
    print("Saving price_by_room_type.png...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_listings[df_listings['price'] < 800], x='room_type', y='price', palette='Set2')
    plt.title("Price Range Comparison by Room Type")
    plt.xlabel("Room Type")
    plt.ylabel("Price (€)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "price_by_room_type.png"), dpi=300)
    plt.close()
    
    # --- CHART 3: Seasonality Line Chart ---
    print("Saving pricing_seasonality.png...")
    plt.figure(figsize=(12, 6))
    plt.plot(df_calendar['date'], df_calendar['avg_price'], color='#ff7f0e', linewidth=2)
    plt.title("Average Nightly Rate Trend Across 365 Days")
    plt.xlabel("Date")
    plt.ylabel("Average Price (€)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pricing_seasonality.png"), dpi=300)
    plt.close()
    
    print(f"🎉 Visualizations saved successfully in '{plot_dir}/'")

if __name__ == "__main__":
    generate_charts()