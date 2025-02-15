# Overview

This project implements a momentum-based trading strategy for the S&P 500 using a Geometric Brownian Motion (GBM) model. It uses historical data, statistical modeling and a SQLite database to backtest a simple trading algorithm.

## Workflow

The following workflow is used:

- Create tables for managing trades and access to data.
- Calibrate a GBM model to the most recent market data.
- Forecast future prices (e.g., 10 days ahead) along with confidence intervals.
- Calculate expected shortfall from the forecast.
- Generate trading signals based on the forecast and risk measures.
- Record positions in a dedicated database table.
- Backtest the strategy over a one-year period (from 2020-06-01 to 2021-05-31).
- Visualize performance by plotting the evolution of total wealth.
