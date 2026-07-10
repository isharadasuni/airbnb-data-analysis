import os
import duckdb
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def cohen_d(x, y):
    """Calculates Cohen's d effect size for two independent groups."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def run_statistical_analysis(db_path="data/processed/airbnb_analytics.db"):
    print("🔬 Running Statistical Analysis & Hypothesis Testing...")
    
    # Connect to the DuckDB database
    con = duckdb.connect(db_path)
    
    # 1. Fetch listings data
    df = con.execute("""
        SELECT listing_price AS price, room_type, accommodates, bedrooms, beds
        FROM dim_listings
        WHERE listing_price IS NOT NULL 
          AND accommodates IS NOT NULL 
          AND bedrooms IS NOT NULL 
          AND beds IS NOT NULL;
    """).df()
    con.close()
    
    # --- STEP 1: HYPOTHESIS TESTING (H1) ---
    print("\n--- H1: Entire Home vs. Private Room Pricing ---")
    entire_homes = df[df['room_type'] == 'Entire Home/Apt']['price'].dropna().values
    private_rooms = df[df['room_type'] == 'Private Room']['price'].dropna().values
    
    if len(entire_homes) > 0 and len(private_rooms) > 0:
        # Run Welch's t-test (equal_var=False)
        t_stat, p_val = stats.ttest_ind(entire_homes, private_rooms, equal_var=False)
        effect_size = cohen_d(entire_homes, private_rooms)
        
        print(f"• Mean Entire Home Price: €{np.mean(entire_homes):.2f}")
        print(f"• Mean Private Room Price: €{np.mean(private_rooms):.2f}")
        print(f"• Welch's t-statistic: {t_stat:.4f}")
        print(f"• p-value: {p_val:.4e}")
        print(f"• Cohen's d Effect Size: {effect_size:.4f}")
        
        if p_val < 0.05:
            print("👉 Decision: Reject Null Hypothesis (H0). The pricing difference is statistically significant.")
        else:
            print("👉 Decision: Fail to Reject Null Hypothesis (H0).")
            
    # --- STEP 2: OLS REGRESSION ---
    print("\n--- OLS Regression (Drivers of Price) ---")
    # Log-transform prices to handle skewness
    df['log_price'] = np.log(df['price'])
    
    # Define Independent Variables (X) and Dependent Variable (y)
    X = df[['accommodates', 'bedrooms', 'beds']]
    X_with_const = sm.add_constant(X) # Statsmodels requires adding a constant intercept
    y = df['log_price']
    
    # Fit OLS Model
    model = sm.OLS(y, X_with_const).fit()
    print(model.summary())
    
    # --- STEP 3: MULTICOLLINEARITY (VIF) ---
    print("\n--- Variance Inflation Factor (VIF) Check ---")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print(vif_data)
    print("\n*(Rule of thumb: VIF < 5.0 indicates no severe multicollinearity issues)*")

if __name__ == "__main__":
    run_statistical_analysis()