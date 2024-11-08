import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools

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


# Train-test split (80% train, 20% test)
train_size = int(len(target) * 0.8)
train_target, test_target = target[:train_size], target[train_size:]
train_exog, test_exog = exog[:train_size], exog[train_size:]

# Fit the SARIMAX model
# Example SARIMAX order (p=1, d=1, q=1), seasonal_order=(P=1, D=1, Q=1, 12 for yearly seasonality)
# sarimax_model = SARIMAX(train_target, exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 10)).fit()

# Define the p, d, q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Seasonal P, D, Q with period=12 (for yearly seasonality)
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None

# Grid search to find the best set of parameters
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(train_target, exog=train_exog, order=param, seasonal_order=param_seasonal)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
            print(f'Current SARIMAX order: {best_pdq}')
            print(f'Current Seasonal order: {best_seasonal_pdq}')
        except Exception as e:
            continue

print(f'Best SARIMAX order: {best_pdq}')
print(f'Best Seasonal order: {best_seasonal_pdq}')

