========================================================================
PROJECT TITLE: AMSTERDAM AIRBNB DATA ENGINEERING & ANALYTICS PIPELINE
========================================================================

Prepared by: Ishara Dasuni Senadheera
GitHub Repository: https://github.com/isharadasuni/airbnb-data-analysis.git
July 2026

------------------------------------------------------------------------

1. OVERVIEW
-----------
This repository implements an automated, production-grade end-to-end data
engineering and predictive machine learning pipeline using public Inside 
Airbnb data for Amsterdam, Netherlands.

The pipeline automates:
- Streaming compressed CSV downloads with progress trackers.
- Preprocessing, validation, bounding filters, and type formatting.
- Relational Star Schema compilation within a local DuckDB OLAP database.
- Hypothesis testing (Welch's t-test) and regression driver models (OLS).
- Supervised price estimation (XGBoost) and property clustering (K-Means).
- Programmatic IEEE-formatted Word Document compilation.
- Interactive dashboard visualization (Streamlit).


2. PROJECT STRUCTURE
--------------------
The directory structure is organized as follows:

airbnb/
├── data/
│   ├── raw/                      <-- Downloads (listings.csv.gz, calendar.csv.gz)
│   └── processed/
│       ├── plots/                <-- Generated PNG plots for reports
│       ├── airbnb_analytics.db   <-- DuckDB columnar OLAP database file
│       ├── model_metrics.json    <-- XGBoost evaluation statistics (MAE, RMSE)
│       └── *.parquet             <-- Cleaned tables (listings, calendar)
├── app.py                        <-- Streamlit interactive dashboard UI
├── ingest.py                     <-- Downloads raw gzipped CSV files
├── clean.py                      <-- Parses floats, handles nulls, saves Parquet
├── transform.py                  <-- Structures DuckDB Star Schema tables
├── eda.py                        <-- Generates distribution & seasonal plots
├── stats.py                      <-- Performs Welch's t-test, OLS, and VIF checks
├── ml.py                         <-- Trains XGBoost model & segments listings
├── report.pdf					   <-- Compiles IEEE report 
└── README.txt                    <-- Project documentation 


3. PREREQUISITES & INSTALLATION
-------------------------------
To run this project, you need Python 3.9+ installed.

Step 1: Install required libraries
Open your terminal/command prompt and run:
pip install pandas numpy duckdb python-docx xgboost streamlit matplotlib seaborn scipy statsmodels scikit-learn

Step 2: Navigate to your workspace directory


4. HOW TO RUN THE PIPELINE STAGES
---------------------------------
Run the scripts sequentially to execute the full data pipeline:

Stage 1: Download raw web streams
> python ingest.py

Stage 2: Clean types, impute missing values, and validate coordinates
> python clean.py

Stage 3: Establish dimensions and fact tables inside DuckDB
> python transform.py

Stage 4: Create exploratory visualization plots
> python eda.py

Stage 5: Run hypothesis significance tests and OLS regression drivers
> python stats.py

Stage 6: Train XGBoost price predictor and compile K-Means clusters
> python ml.py

Stage 7: Launch the interactive Streamlit dashboard
> streamlit run app.py


5. PIPELINE FEATURES & CONSTRAINTS
----------------------------------
- Memory Optimization: Ingest and cleaning streams download in chunks to 
  prevent local memory Out-Of-Memory (OOM) failures.
- Star Schema: Relational tables mapped into 'dim_listings', 'dim_hosts', 
  and 'fact_calendar' for sub-second analytical processing.
- Strict Integers: The Streamlit pricing sandbox restricts capacity 
  inputs (Bedrooms, Beds, Accommodates) to positive integers.


6. LICENSING & SYSTEM CONFIGURATION
-----------------------------------
All data streams originate from Inside Airbnb. Model weights are serialized 
locally on execution in 'xgb_price_model.pkl'.



========================================================================
