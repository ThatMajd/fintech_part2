import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

# Load the dataset
data_path = 'fintech_part2//combined_ticks.csv'
df = pd.read_csv(data_path)

print('**************')

# df.drop(columns=['Iron_Close'], inplace=True)

# Linear interpolation for filling missing values
df = df.interpolate(method='linear', inplace=False)

# Forward fill for any remaining NaNs at the beginning
df = df.ffill()

# Backward fill for any remaining NaNs at the end
df = df.bfill()

print('**************')

# Fill null values with the average of previous and next rows
# df = df.fillna((df.shift() + df.shift(-1)) / 2, inplace=True)

# Delete rows with null values
# df.dropna(inplace=True)

print('**************')

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

# Initialize Prophet model
model = Prophet()

regressors = features.copy()
regressors.remove('y')
regressors.remove('ds')

for item in regressors:
    # print(item)
    # print(df[item].isnull().sum())
    if item =='y' or item == 'ds':
        continue
    model.add_regressor(item)

# Add additional regressors
# model.add_regressor('feature1')
# model.add_regressor('feature2')

# Fit the model
model.fit(df)

# Create a DataFrame for future predictions
# future = model.make_future_dataframe(periods=30)

# # For simplicity, let's use the last 30 rows of df as future regressor data (replace with your actual future data)
# future_regressor_data = df[regressors].tail(30).reset_index(drop=True)
# future = future.merge(future_regressor_data, left_index=True, right_index=True)

# print('future')
# print(future)

# # # Add the same regressors to the future DataFrame
# # future['feature1'] = range(100, 130)
# # future['feature2'] = range(200, 230)

# # Make predictions
# forecast = model.predict(future)

# # Plot the forecast
# model.plot(forecast)
# plt.show()

# # Plot the components (including the effect of each regressor)
# model.plot_components(forecast)
# plt.show()

# Create a DataFrame for the full date range (past + future)
full_date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max() + pd.Timedelta(days=30), freq='D')
full_df = pd.DataFrame({'ds': full_date_range})

full_df = pd.merge(full_df, df[regressors + ['ds']], on='ds', how='left')

# Add future regressor data (assuming you have this data prepared)
# For simplicity, use the last available values for future dates as a placeholder
# Replace this with actual future data if available

for regressor in regressors:
    full_df[regressor].fillna(method='ffill', inplace=True)  # Forward fill with last available values

# Make predictions on the full date range
forecast = model.predict(full_df)

# Calculate prediction accuracy
accuracy = 1 - (abs(forecast['yhat'] - df['y']) / df['y'])
average_accuracy = accuracy.mean()
print('Prediction Accuracy:', average_accuracy)


mape = np.mean(np.abs((df['y'] - forecast['yhat']) / df['y'])) * 100
print('Mean Absolute Percentage Error (MAPE):', mape)


# View the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Plot the forecast
model.plot(forecast)
plt.savefig('forecast_plot.png')
plt.legend()
plt.show()

# Plot the components (including the effect of each regressor)
# model.plot_components(forecast)
# plt.savefig('forecast_reg_plot.png')
# plt.show()
