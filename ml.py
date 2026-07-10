import os
import duckdb
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def run_machine_learning(db_path="data/processed/airbnb_analytics.db", output_dir="data/processed"):
    print("🤖 Starting Machine Learning modeling stage...")
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Connect to the DuckDB database
    con = duckdb.connect(db_path)
    
    # 1. Fetch features for model training
    df = con.execute("""
        SELECT 
            listing_price AS price,
            latitude,
            longitude,
            accommodates,
            bedrooms,
            beds,
            occupancy_rate,
            estimated_revenue
        FROM dim_listings
        WHERE listing_price IS NOT NULL 
          AND accommodates IS NOT NULL 
          AND bedrooms IS NOT NULL 
          AND beds IS NOT NULL
          AND occupancy_rate IS NOT NULL
          AND estimated_revenue IS NOT NULL;
    """).df()
    con.close()
    
    # Ensure inputs like accommodates, bedrooms, and beds are parsed as integers
    df['accommodates'] = df['accommodates'].astype(int)
    df['bedrooms'] = df['bedrooms'].astype(int)
    df['beds'] = df['beds'].astype(int)
    
    # -----------------------------------------------------------------
    # PART A: XGBOOST PRICE PREDICTION
    # -----------------------------------------------------------------
    print("\n--- Training XGBoost Pricing Regressor ---")
    
    # Prepare features (X) and target (y)
    # Predict price using location coordinates and capacity attributes
    feature_cols = ['latitude', 'longitude', 'accommodates', 'bedrooms', 'beds']
    X = df[feature_cols]
    y = df['price'].values
    y_log = np.log(y) # Train on log price to manage right-skewness
    
    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    
    print("Running 5-Fold Cross-Validation...")
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_log, y_val = y_log[train_idx], y[val_idx]
        
        # Train model
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_log)
        
        # Predict and reverse the log transformation
        pred_log = model.predict(X_val)
        pred = np.exp(pred_log)
        
        # Calculate Mean Absolute Error (MAE)
        maes.append(mean_absolute_error(y_val, pred))
        
    print(f"👉 Average Cross-Validation MAE: €{np.mean(maes):.2f}")
    
    # Train the final model on all data
    final_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    final_model.fit(X, y_log)
    
    # Save the model and feature list for our dashboard
    with open(os.path.join(output_dir, "models", "xgb_price_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    with open(os.path.join(output_dir, "models", "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print("Saved final model to data/processed/models/xgb_price_model.pkl")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    importances = final_model.feature_importances_
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    feat_series.plot(kind='barh', color='#8dd3c7')
    plt.title("Price Predictor Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "feature_importance.png"), dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # PART B: K-MEANS LISTING CLUSTERING
    # -----------------------------------------------------------------
    print("\n--- Running K-Means Listing Segmentation ---")
    
    # Select columns to cluster on
    cluster_cols = ['price', 'accommodates', 'occupancy_rate', 'estimated_revenue']
    
    # Standardize features (forces mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_cols])
    
    # Fit KMeans with K=3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Compute profiles for each cluster
    cluster_means = df.groupby('cluster')[cluster_cols].mean()
    cluster_counts = df['cluster'].value_counts()
    
    # Sort cluster IDs by average price to label them logically
    sorted_idx = cluster_means['price'].sort_values().index.tolist()
    labels = {
        sorted_idx[0]: "Budget Segment",
        sorted_idx[1]: "Mid-Tier Entire Homes",
        sorted_idx[2]: "Premium / Commercial Listings"
    }
    
    profiles = {}
    for c_id in range(3):
        label_name = labels[c_id]
        profiles[label_name] = {
            "count": int(cluster_counts[c_id]),
            "mean_price": float(cluster_means.loc[c_id, 'price']),
            "mean_accommodates": float(cluster_means.loc[c_id, 'accommodates']),
            "mean_occupancy_rate": float(cluster_means.loc[c_id, 'occupancy_rate']),
            "mean_estimated_revenue": float(cluster_means.loc[c_id, 'estimated_revenue'])
        }
        
    # Save profiles to JSON file for dashboard / reporting
    with open(os.path.join(output_dir, "clustering_results.json"), "w") as f:
        json.dump(profiles, f, indent=4)
        
    print("Listing segment profiles:")
    print(json.dumps(profiles, indent=4))
    
    # Plot clustering chart
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df['price'] < 800], x='price', y='occupancy_rate', hue='cluster', palette='Set1', alpha=0.6)
    plt.title("K-Means Listings Clusters (Amsterdam)")
    plt.xlabel("Price (€)")
    plt.ylabel("Occupancy Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "listing_clusters.png"), dpi=300)
    plt.close()
    
    print("\n🎉 ML modeling and plot generation finished successfully!")

if __name__ == "__main__":
    run_machine_learning()