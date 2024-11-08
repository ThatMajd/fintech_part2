import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset (assuming 'Gold_Close' is the target and other columns are exogenous variables)
data = pd.read_csv('combined_ticks.csv', index_col=0, parse_dates=True)

data = data.asfreq('D')

# Linear interpolation for filling missing values
data = data.interpolate(method='linear', inplace=False)

# Forward fill for any remaining NaNs at the beginning
data = data.ffill()

# Backward fill for any remaining NaNs at the end
data = data.bfill()



# Define the target variable (Gold_Close) and the exogenous variables (all other columns)
target = data['Gold_Close']
exog = data.drop(columns=['Gold_Close'])

print(f'exogenous ==> {exog}')

# Plot the target variable
plt.plot(target)
plt.title('Gold_Close Prices')
plt.savefig('gold_close_prices_sarimax.png')
plt.clf()

# Train-test split (80% train, 20% test)
train_size = int(len(target) * 0.8)
train_target, test_target = target[:train_size], target[train_size:]
train_exog, test_exog = exog[:train_size], exog[train_size:]

# Fit the SARIMAX model
# Example SARIMAX order (p=1, d=1, q=1), seasonal_order=(P=1, D=1, Q=1, 12 for yearly seasonality)
sarimax_model = SARIMAX(train_target, exog=train_exog, order=(0, 1, 2), seasonal_order=(1, 0, 1, 7)).fit()

# Forecast future values (for the length of the test set)
sarimax_forecast = sarimax_model.forecast(steps=len(test_target), exog=test_exog)

# Plot the forecast against the actual values
plt.plot(train_target.index, train_target, label='Training Data')
plt.plot(test_target.index, test_target, label='Test Data')
plt.plot(test_target.index, sarimax_forecast, label='SARIMAX Forecast')
plt.title('SARIMAX Forecast with Exogenous Variables')
plt.legend()
plt.savefig('sarimax_forecast.png')
plt.clf()

# Evaluate the model with RMSE
rmse_sarimax = np.sqrt(mean_squared_error(test_target, sarimax_forecast))
print(f'SARIMAX RMSE: {rmse_sarimax}')
