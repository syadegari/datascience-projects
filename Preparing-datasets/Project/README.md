
# ML Pipeline for Feature Engineering

This demonstrates an example of complete data engineering pipeline for a machine learning project in the context of financial data analysis. The goal is to ingest, clean, and preprocess data for subsequent machine learning tasks. Some macroeconomic indicators and historical stock prices are used and tasks such as data cleaning, imputation, resampling, normalization, and exploratory data analysis (EDA) are performed.

## Overview

The script performs the following tasks:

- **Data Ingestion:**  
  Import data from CSV files hosted on a public GitHub repository.  
  *Files used:*
  - GDP data (`GDP.csv`)
  - Inflation data (`inflation_monthly.csv`)
  - Apple stock prices (`apple_historical_data.csv`)
  - Microsoft stock prices (`microsoft_historical_data.csv`)

- **Data Exploration:**  
  Verify data integrity using methods like `.head()`, `.info()`, and `.describe()`.

- **Data Preprocessing:**  
  - Check and impute missing data (using forward fill for Apple's missing prices).
  - Remove special characters from numeric columns and cast them to numeric types.
  - Convert date strings to `datetime` objects and set them as indices.
  - Align datetime data (adjust inflation data to the month end).
  - Resample data: downsample monthly inflation to quarterly, and upsample to weekly with interpolation.
  - Normalize/standardize GDP data using `StandardScaler`.

- **Exploratory Data Analysis (EDA):**  
  - Plot time series comparing Apple's open and close prices (for the last three months).
  - Create a histogram of Apple's closing prices over the same period.
  - Calculate stock returns and merge them with inflation changes.
  - Plot a correlation heatmap between Apple, Microsoft, and inflation.
  - Calculate and plot the weekly rolling volatility (standard deviation) of Apple's closing price alongside the price itself using dual y-axes.

- **Exporting Data:**  
  Save the processed data to new CSV files for future use, including:
  - Weekly and quarterly inflation data
  - Modified Apple and Microsoft stock datasets
  - A merged dataset of monthly percentage changes for Apple, Microsoft, and inflation

## Dependencies

The script uses the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

