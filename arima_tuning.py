from pycaret.time_series import *
import pandas as pd

# Load the data with date as index
df = pd.read_csv('combined_ticks.csv', parse_dates=True, index_col=0)

# Check the first few rows
print(df.head())

print(df.index.freq)
df = df.asfreq('D')
print(df.index.freq)

# Initialize PyCaret setup for time series forecasting
exp = setup(data = df, 
            target='Gold_Close',   # Target column (what you're predicting)
            session_id=123,        # For reproducibility
            fold=1,                # Number of folds for cross-validation
            fh=12,
            n_jobs = -1,
            numeric_imputation_target='mean',
            numeric_imputation_exogenous='mean')              # Forecast horizon

auto_arima_model = create_model('auto_arima')

# Step 3: Save the trained model to a pkl file
save_model(auto_arima_model, 'auto_arima_model')

# Step 4: Tune the Auto-ARIMA model (optional)
tuned_auto_arima = tune_model(auto_arima_model)

# Step 5: Save the tuned model to a pkl file (optional)
save_model(tuned_auto_arima, 'tuned_auto_arima_model')