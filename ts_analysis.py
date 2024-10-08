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
            numeric_imputation_target='mean',
            numeric_imputation_exogenous='mean')              # Forecast horizon

# Compare models
best_model = compare_models()

# Get the comparison results in a DataFrame
comparison_results = pull()

# Save the DataFrame to a CSV file
comparison_results.to_csv('model_comparison_results.csv', index=False)

# best_model_arima = create_model('auto_arima')