import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = 'data//combined_ticks.csv'
# data_path = 'combined_ticks.csv'
df = pd.read_csv(data_path)

train_size = 0.8
test_size = 1-train_size

# print('**************')

# df.drop(columns=['Iron_Close'], inplace=True)

# Linear interpolation for filling missing values
df = df.interpolate(method='linear', inplace=False)

# Forward fill for any remaining NaNs at the beginning
df = df.ffill()

# Backward fill for any remaining NaNs at the end
df = df.bfill()

print(df.info())
print("Shape: ", df.shape)

# print('**************')

# Fill null values with the average of previous and next rows
# df = df.fillna((df.shift() + df.shift(-1)) / 2, inplace=True)

# Delete rows with null values
# df.dropna(inplace=True)

# print('**************')

# print(df['Unnamed: 0'])
# Rename column 'Gold_Close' to 'y'
df.rename(columns={'Gold_Close': 'y'}, inplace=True)
df.rename(columns={'Unnamed: 0': 'ds'}, inplace=True)

# Convert 'ds' column to datetime format
df['ds'] = pd.to_datetime(df['ds'])
# print(df.isnull().sum())

print(df.info())
# List all features
features = df.columns.tolist()
print("Features:", features)

print("Shape: ", df.shape)

# Initialize Prophet model
model = Prophet()

regressors = features.copy()
regressors.remove('y')
regressors.remove('ds')

# Add regressors to the model
for item in regressors:
    # print(item)
    # print(df[item].isnull().sum())
    if item =='y' or item == 'ds':
        continue
    model.add_regressor(item)

# Add additional regressors
# model.add_regressor('feature1')

# Create a DataFrame for the full date range (past + future)
length = len(df)

# full_df = pd.merge(full_df, df[regressors + ['ds']], on='ds', how='left')
date_range = pd.date_range(end=df['ds'].max(), periods=length, freq='D')
full_df = df.copy()
full_df['ds'] = date_range

# Save full_df to a CSV file
full_df.to_csv('data//combined_for_ts.csv')

# Fit the model
model.fit(full_df)

# print('Full DataFrame Columns Check =>',full_df.columns)

# Make predictions on the full date range
forecast = model.predict(full_df)

# Calculate prediction accuracy
accuracy = 1 - (abs(forecast['yhat'] - df['y']) / df['y'])
average_accuracy = accuracy.mean()
print('Average Prediction Accuracy:', average_accuracy)

# Calculate the Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((df['y'] - forecast['yhat']) / df['y'])) * 100
print('Mean Absolute Percentage Error (MAPE):', mape)

# Calculate the R-squared value
r2 = r2_score(df['y'], forecast['yhat'])
print('R-squared:', r2)


# View the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Plot the forecast
model.plot(forecast)
plt.savefig('prophet_forecast_plot.png')
plt.legend()
# plt.show()

comparison_df = forecast.tail(int(test_size * (len(forecast))))

# print("Columns =>",comparison_df.columns)
# Rename columns for comparison
comparison_df = comparison_df.rename(columns={'yhat': 'Predicted'})

# Select only the 'actual' and 'predicted' columns
comparison_df = comparison_df[['Predicted']]

comparison_df['Actual'] = df.tail(int(test_size*(len(forecast))))['y']

# Remove index from the DataFrame
comparison_df.reset_index(drop=True, inplace=True)

# Calculate the Accuracy
accuracy = 1 - (abs(comparison_df['Actual'] - comparison_df['Predicted']) / comparison_df['Actual'])
average_accuracy = accuracy.mean()
print('Average Prediction Accuracy on Test:', average_accuracy)
# Calculate the Mean Absolute Percentage Error (MAPE)
mape = np.mean(abs(comparison_df['Actual'] - comparison_df['Predicted']) / comparison_df['Actual'])
print('Mean Absolute Percentage Error (MAPE) on Test:', mape)
# Calculate the R-squared value
r2 = r2_score(comparison_df['Actual'], comparison_df['Predicted'])
print('R-squared on Test:', r2)


# Display the comparison DataFrame
# print(comparison_df)

comparison_df.to_csv('prophet_comparison.csv')


# Buy - Sell Signal
# print("Forecast Type")
# print(type(forecast))

# forecast = forecast.tail(35)

# # Calculate percentage change between consecutive days
# forecast['PriceChange'] = forecast['yhat'].pct_change()

# Define Buy/Sell signals based on percentage price changes
# You can adjust the threshold for defining buy/sell conditions

# Buy if predicted price increases by more than 1% compared to the previous day
# forecast['Signal'] = np.where(forecast['PriceChange'] > 0.01, 'Buy', 
#                   np.where(forecast['PriceChange'] < -0.01, 'Sell', 'Hold'))

# Check the signals
# print(forecast)

# # Backtest strategy
# # In backtesting, simulate trades based on the "Buy", "Sell", and "Hold" signals generated from the predicted prices.
# # Assume we start with an initial capital of $10,000 and no position
# initial_capital = 10000
# position = 0
# capital = initial_capital

# portfolio_values = []  # To store the value of the portfolio over time
# positions = []  # To track the positions held over time

# capital = 10000  # Starting capital
# position = 0  # Initial position

# # Assume 'forecast' is your DataFrame that includes 'yhat' (predicted price) and 'Signal' (buy/sell signal)

# # Shift the Signal column back by 1 day to avoid reacting late to the changes
# forecast['Signal'] = forecast['Signal'].shift(-1)

# # Remove the last row because the shifted signal will introduce a NaN value
# forecast = forecast[:-1]

# # Simulate trading
# for index, row in forecast.iterrows():
#     if row['Signal'] == 'Buy' and position == 0:  # Buy only if no position
#         position = capital / row['yhat']  # Buy as many units as possible with current capital
#         capital = 0  # Use all capital to buy the product
#     elif row['Signal'] == 'Sell' and position > 0:  # Sell if we hold a position
#         capital = position * row['yhat']  # Sell all units at the current price
#         position = 0  # Clear the position after selling

#     # Track the current position
#     positions.append(position)

#     # Calculate portfolio value: if position > 0, value is position * price; else, it's just capital
#     if position > 0:
#         portfolio_value = position * row['yhat']
#     else:
#         portfolio_value = capital

#     portfolio_values.append(portfolio_value)

# # Store portfolio values in the forecast DataFrame
# forecast['PortfolioValue'] = portfolio_values

# forecast['PriceChange'].fillna(0, inplace=True)

# # Debugging output to check the final portfolio value
# print(forecast[['yhat', 'Signal', 'PortfolioValue']])

# # Calculate daily returns of the portfolio
# forecast['PortfolioReturn'] = forecast['PortfolioValue'].pct_change()

# # Insert 0 where PortfolioReturn is null
# forecast['PortfolioReturn'].fillna(0, inplace=True)

# print(forecast[['yhat', 'Signal', 'PortfolioValue','PortfolioReturn']])

# # Risk-free rate (e.g., 0.01 for 1% per year, adjusted for daily returns)
# # risk_free_rate = 0.01 / 252  # Assuming 252 trading days in a year

# # # Calculate the Sharpe Ratio
# # excess_returns = forecast['PortfolioReturn'] - risk_free_rate
# # sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
# # print(f'Sharpe Ratio: {sharpe_ratio}')

# # # Calculate maximum drawdown
# # forecast['CumulativeReturn'] = (1 + forecast['PortfolioReturn']).cumprod()
# # forecast['CumulativeMax'] = forecast['CumulativeReturn'].cummax()
# # forecast['Drawdown'] = (forecast['CumulativeReturn'] - forecast['CumulativeMax']) / forecast['CumulativeMax']
# # max_drawdown = forecast['Drawdown'].min()
# # print(f'Maximum Drawdown: {max_drawdown}')

# # Portfolio returns (replace with actual returns)
# portfolio_returns = forecast['PortfolioReturn']

# # 1. Sharpe Ratio
# risk_free_rate = 0.01 / 252  # Daily risk-free rate
# excess_returns = portfolio_returns - risk_free_rate
# sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / portfolio_returns.std())
# print(f'Sharpe Ratio: {sharpe_ratio}')

# # 2. Volatility (Annualized)
# volatility = portfolio_returns.std() * np.sqrt(252)
# print(f'Annualized Volatility: {volatility}')

# # 3. Maximum Drawdown
# cumulative_return = (1 + portfolio_returns).cumprod()
# cumulative_max = cumulative_return.cummax()
# drawdown = (cumulative_return - cumulative_max) / cumulative_max
# max_drawdown = drawdown.min()
# print(f'Maximum Drawdown: {max_drawdown}')