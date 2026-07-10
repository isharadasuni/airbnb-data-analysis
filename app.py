import os
import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import pickle
import json

# 1. Page Configuration for premium aesthetics
st.set_page_config(
    page_title="Amsterdam Airbnb Market Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database path
DB_PATH = "data/processed/airbnb_analytics.db"

@st.cache_resource
def get_db_connection():
    if os.path.exists(DB_PATH):
        return duckdb.connect(DB_PATH, read_only=True)
    return None

con = get_db_connection()

# Load ML Components
MODEL_PATH = "data/processed/models/xgb_price_model.pkl"
FEATURE_PATH = "data/processed/models/feature_cols.json"
xgb_model = None
feature_cols = []

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
if os.path.exists(FEATURE_PATH):
    with open(FEATURE_PATH) as f:
        feature_cols = json.load(f)

# Sidebar Navigation
st.sidebar.title("🏠 Airbnb Market Intelligence")
st.sidebar.caption("Learning Dashboard")
page = st.sidebar.radio(
    "Navigation", 
    ["Market Overview & KPIs", "Spatial & Location Analysis", "Listing Segmentation", "Price Prediction Sandbox"]
)

if con is None:
    st.error("Database not found! Please run your scripts (`ingest.py`, `clean.py`, `transform.py`) first to build it.")
else:
    # 2. Extract High-Level KPIs
    kpi_df = con.execute("""
        SELECT 
            COUNT(*) as total_listings,
            ROUND(AVG(listing_price), 2) as avg_price,
            ROUND(AVG(occupancy_rate) * 100, 2) as avg_occupancy,
            ROUND(AVG(estimated_revenue), 2) as avg_revenue
        FROM dim_listings;
    """).df()
    
    total_hosts = con.execute("SELECT COUNT(DISTINCT host_id) FROM dim_listings;").fetchone()[0]

    # =================================================================
    # PAGE 1: MARKET OVERVIEW & KPIS
    # =================================================================
    if page == "Market Overview & KPIs":
        st.title("🏠 Amsterdam Airbnb Market Overview")
        st.markdown("Dynamic metrics and seasonality trends derived from short-term rental listings.")
        
        # Display Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Listings", f"{kpi_df['total_listings'].values[0]:,}")
        col2.metric("Unique Hosts", f"{total_hosts:,}")
        col3.metric("Average Nightly Rate", f"€{kpi_df['avg_price'].values[0]:,.2f}")
        col4.metric("Avg. Occupancy Rate", f"{kpi_df['avg_occupancy'].values[0]:.1f}%")
        col5.metric("Avg. Est. Annual Revenue", f"€{kpi_df['avg_revenue'].values[0]:,.2f}")
        
        # Plotly Visualizations in columns
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Price Distribution by Room Type")
            room_df = con.execute("""
                SELECT room_type, listing_price as price
                FROM dim_listings
                WHERE listing_price < 800;
            """).df()
            fig = px.box(
                room_df, x="room_type", y="price", color="room_type",
                title="Nightly Prices by Room Type (< €800)",
                labels={"room_type": "Room Type", "price": "Price (€)"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Pricing Seasonality Trend")
            seas_df = con.execute("""
                SELECT date, AVG(price) as avg_price
                FROM fact_calendar
                GROUP BY date
                ORDER BY date;
            """).df()
            fig_seas = px.line(
                seas_df, x="date", y="avg_price",
                title="Average Nightly Price over 365 Days",
                labels={"date": "Date", "avg_price": "Average Price (€)"}
            )
            fig_seas.update_traces(line_color='#ff7f0e')
            st.plotly_chart(fig_seas, use_container_width=True)

    # =================================================================
    # PAGE 2: SPATIAL & LOCATION ANALYSIS
    # =================================================================
    elif page == "Spatial & Location Analysis":
        st.title("🗺️ Geographic Density Map")
        st.markdown("Interactive map showing listing distribution and locations across Amsterdam.")
        
        # Load coordinate data
        geo_df = con.execute("""
            SELECT latitude, longitude, listing_price as price, room_type
            FROM dim_listings
            WHERE listing_price < 1000;
        """).df()
        
        # Budget Slider
        price_range = st.slider("Select Price Range Filter (€)", 0, 1000, (50, 400))
        filtered_geo = geo_df[geo_df['price'].between(price_range[0], price_range[1])]
        
        # Display coordinate map
        st.map(filtered_geo[['latitude', 'longitude']], size=15)
        
        # Neighborhood Table
        st.subheader("Top Neighbourhoods Ranked by Nightly Rate")
        neigh_df = con.execute("""
            SELECT 
                neighbourhood,
                COUNT(*) as listing_count,
                ROUND(AVG(listing_price), 2) as avg_price,
                ROUND(AVG(occupancy_rate) * 100, 2) as average_occupancy_rate,
                ROUND(AVG(estimated_revenue), 2) as average_annual_revenue
            FROM dim_listings
            GROUP BY neighbourhood
            ORDER BY avg_price DESC;
        """).df()
        st.dataframe(neigh_df, use_container_width=True)

    # =================================================================
    # PAGE 3: LISTING SEGMENTATION (CLUSTERING)
    # =================================================================
    elif page == "Listing Segmentation":
        st.title("📊 Listing Segmentation (K-Means Clustering)")
        st.markdown("Segments listings into distinct commercial tiers using mathematical grouping algorithms.")
        
        # Load cluster results JSON
        cluster_json_path = "data/processed/clustering_results.json"
        if os.path.exists(cluster_json_path):
            with open(cluster_json_path) as f:
                c_data = json.load(f)
            
            # Draw Columns for Cluster Summaries
            cols = st.columns(3)
            for idx, (label, details) in enumerate(c_data.items()):
                with cols[idx]:
                    st.info(f"### {label}")
                    st.metric("Total Properties", f"{details['count']:,}")
                    st.write(f"• **Average Nightly Rate:** €{details['mean_price']:.2f}")
                    st.write(f"• **Average Capacity:** {details['mean_accommodates']:.1f} guests")
                    st.write(f"• **Average Occupancy Rate:** {details['mean_occupancy_rate']*100:.1f}%")
                    st.write(f"• **Average Annual Revenue:** €{details['mean_estimated_revenue']:.2f}")
        else:
            st.warning("Clustering profiles data file not found! Run ml.py first.")
            
        # Display Clustering Plot
        plot_path = "data/processed/plots/listing_clusters.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Listing Segmentation scatterplot", use_container_width=True)

    # =================================================================
    # PAGE 4: PRICE PREDICTION SANDBOX (WITH INTEGRITY CORRECTIONS)
    # =================================================================
    elif page == "Price Prediction Sandbox":
        st.title("🔮 AI Price Prediction Sandbox")
        st.markdown("Recommended nightly pricing generated by the trained XGBoost regression model.")
        
        if xgb_model is None:
            st.warning("Pricing Model file not found! Run ml.py first.")
        else:
            c1, c2 = st.columns(2)
            
            with c1:
                # 👉 CORRECTION: Strict positive integer fields using Streamlit value rules
                accommodates = st.number_input(
                    "Accommodates (Max Guests)",
                    min_value=1,
                    max_value=16,
                    value=2,
                    step=1  # Forces strict integer inputs
                )
                
                bedrooms = st.number_input(
                    "Number of Bedrooms",
                    min_value=1,
                    max_value=10,
                    value=1,
                    step=1  # Strict positive integer input control
                )
                
                beds = st.number_input(
                    "Number of Beds",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1  # Strict positive integer input control
                )
                
            with c2:
                # Coordinate inputs
                latitude = st.number_input("Latitude Coordinate", 52.0, 53.0, 52.3702, format="%.4f")
                longitude = st.number_input("Longitude Coordinate", 4.5, 5.5, 4.8952, format="%.4f")
                
            # Compile features (matching model features from ml.py)
            # Schema: ['latitude', 'longitude', 'accommodates', 'bedrooms', 'beds']
            input_df = pd.DataFrame([{
                'latitude': float(latitude),
                'longitude': float(longitude),
                'accommodates': int(accommodates),
                'bedrooms': int(bedrooms),
                'beds': int(beds)
            }])
            
            # Predict
            pred_log = xgb_model.predict(input_df)[0]
            predicted_price = np.exp(pred_log)
            
            st.success(f"### Recommended Nightly Price: **€{predicted_price:.2f}**")
            st.caption("Pricing predictions are log-transformed estimates calibrated against Amsterdam coordinates.")