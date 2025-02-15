# Stock Trading with Reinforcement Learning 

This is an application of a Deep Q-Network (DQN) reinforcement learning agent for trading strategy using historical stock data. The agent actions are buy, sell, or hold based on historical features such as the adjusted close price and Bollinger Bands.

Here is the high level overview/flow of the program:

1. Data Loading & Exploration

    - Import historical stock and perform exploratory data analysis (EDA) by displaying and plotting the data.

1. Data Cleaning & Feature Engineering

    - Handle missing values using forward fill and compute a 20-day moving average and Bollinger Bands.
    - Select relevant features (Adjusted Close, BB-upper, BB-lower).

1. Normalization and Test/Train Split

    - Normalize the features using sklearn's `StandardScaler` for consistent training.
    - Keep track of the scalers to later reverse the normalization for price interpretation.
    - Split the dataset into training and testing sets.

1. Agent, Model Definition and Training

    - Build a DQN model using Keras.
    - Define an epsilon-greedy agent combined with experience replay buffer.
    - Train the agent over multiple episodes.

1. Testing the Trained Agent

    - Load the saved model and run it on the test data.
    - Generate buy/sell signals and compute performance metrics.
    - Visualize the trading behavior and training losses.
