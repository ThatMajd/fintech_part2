import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load the dataset (assuming 'Gold_Close' is the target and other columns are exogenous variables)
# source = 'data//combined_ticks.csv'
source = 'data//combined_for_ts.csv'
data = pd.read_csv(source)

# Convert the ds column to datetime format
data['ds'] = pd.to_datetime(data['ds'])

# declare ds as index
data.set_index('ds', inplace=True)

data = data.asfreq('D')

print(data.info())


# Define the target variable (Gold_Close) and the exogenous variables (all other columns)
target = data['y']
exog = data.drop(columns=['y','Unnamed: 0'])

print(f'exogenous ==> {exog}')

# Plot the target variable
plt.plot(target)
plt.title('Gold_Close Prices')
plt.savefig('gold_close_prices_sarimax.png')
plt.clf()

# Train-test split
train = 0.8
train_size = int(len(target) * train)
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

# Evaluate the model with AIC
aic_sarimax = sarimax_model.aic
print(f'SARIMAX AIC: {aic_sarimax}')

# Evaluate the model with BIC
bic_sarimax = sarimax_model.bic
print(f'SARIMAX BIC: {bic_sarimax}')

# create a comparison dataframe for the actual and forecasted values
comparison_df = pd.DataFrame({'Actual': test_target, 'Predicted': sarimax_forecast})
# Reset the index to include sequential numbers instead of dates
comparison_df.reset_index(drop=True, inplace=True)
comparison_df.to_csv('sarimax_comparison.csv')

