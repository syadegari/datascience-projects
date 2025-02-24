This project demonstrates how to integrate data acquisition, preprocessing, feature engineering, model tuning and evaluation to build a machine learning model aimed at predicting the future price movement direction of a financial instrument (here XLV)

# Overview

The goal of this project is not to create a foolproof trading strategy but to provide a hands-on experience with:

- Data Acquisition & Cleaning: Downloading and inspecting historical data for XLV, the volatility index (VIX), and Google Trends search interest for "recession."

- Feature Engineering: Creating meaningful features such as cyclical representations of time (month, weekday), historical returns, technical indicators (IBS, Bollinger Bands, RSI), and log-transformed volume.
- Target Definition: Formulating a binary target variable that indicates whether the forward-looking 5-day return is positive or negative.
- Train/Validation/Test Split: Splitting the data temporally to mimic real-world backtesting.
- Model Training and Hyperparameter Tuning: Using a RandomForestClassifier along with learning curves and grid search cross-validation to tune hyperparameters.
- Model Evaluation: Comparing the model’s performance (accuracy, precision, recall, F1-score) against a baseline and examining feature importance for further improvements.
