# Making Sense of Year-on-Year Seasonal Data Using the STL Method with Python

This Python script demonstrates how to use the Seasonal Decomposition of Time Series (STL) method to analyze year-on-year seasonal data. By leveraging the Statsmodels library, it helps to identify trends and anomalies in time series data.

## Features

- Read time series data from a CSV file.
- Fill gaps in the data with zeros.
- Perform STL decomposition on the data to extract trend, seasonal, and residual components.
- Compare the current year's data to the previous year's data.
- Identify anomalies in the data.
- Visualize the results using Matplotlib.

## Requirements

- Python 3.6+
- pandas
- numpy
- statsmodels
- matplotlib

## Usage
```
python yoy.py data_file [-d lookback] [-m] [-v value_column]
```

- `data_file`: name of the CSV file with the time series data.
- `-d`, `--lookback`: number of days of lookback for trend analysis (default: 90).
- `-m`, `--show-model`: show the STL model.
- `-v`, `--value-column`: specify the column name for the value to analyze (default: "value").

## Example

```
python yoy.py example_data.csv -d 90 -m -v value
```

This command will read the time series data from `example_data.csv`, analyze the last 90 days of data, show the STL model, and use the "value" column for analysis.
