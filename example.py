import numpy as np
import pandas as pd
import datetime

# Define parameters
n_points = 365 * 2
trend_slope = .01
seasonal_period = 7
seasonal_amplitude = 10
noise_std = 5

# Generate a date range
start_date = datetime.date.today() - datetime.timedelta(days=365 * 2)
date_range = pd.date_range(start=start_date, periods=n_points, freq='D')

# Generate linear trend
trend = np.arange(n_points) * trend_slope

# Generate seasonal component
seasonal = seasonal_amplitude * \
    np.sin(2 * np.pi * np.arange(n_points) / seasonal_period)

# Generate random noise
np.random.seed(42)  # Set seed for reproducibility
noise = np.random.normal(0, noise_std, n_points)

# Combine trend, seasonal component, and noise
time_series = trend + seasonal + noise

# Create a DataFrame
data = pd.DataFrame({'date': date_range, 'value': time_series})

# Save data to CSV
data.to_csv('example_data.csv', index=False)

print("Example data generated and saved to 'example_data.csv'")
