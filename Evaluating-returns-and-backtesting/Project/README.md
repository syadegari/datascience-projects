# Risk-Parity Portfolio Analysis

## Introduction
This project shows how to construct and evaluate a risk-parity portfolio. Risk-parity is an important strategy since it seeks to balance the rish contribution from each asset into the portfolio.


## Workflow
- Download historical financial data for various assets using the `yfinance` library.
- Resample daily data to monthly frequency to smooth out short-term fluctuations.
- Clean and prepare the data by handling missing values and subsetting relevant columns.
- Calculate arithmetic returns for the assets.
- Compute rolling volatility and derive risk-parity weights based on inverse volatility.
- Apply these weights to calculate the portfolio's weighted returns.
- Evaluate the portfolio using key performance metrics such as annualized return, volatility, Sharpe ratio, Sortino ratio, and maximum drawdown.
- Visualize the cumulative returns and drawdowns over time.
