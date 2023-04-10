import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import WeekdayLocator, MO
from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
from typing import Tuple, List


def generate_date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Generates a date range between `start_date` and `end_date`.

    Args:
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        pd.DatetimeIndex: A pandas DatetimeIndex representing the date range.
    """
    return pd.date_range(start=start_date, end=end_date, freq='D')


def merge_with_date_range(df: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Merges `df` with a continuous `date_range`.

    Args:
        df (pd.DataFrame): The data to merge with the date range.
        date_range (pd.DatetimeIndex): The continuous date range.

    Returns:
        pd.DataFrame: The merged data.
    """
    continuous_data = pd.DataFrame({'date': date_range})
    continuous_data['date'] = pd.DatetimeIndex(continuous_data['date']).date
    merged_data = continuous_data.merge(
        df, on='date', how='left').fillna(0)
    return merged_data


def fill_gaps_with_zeros(data: pd.DataFrame, start_date: datetime.date, end_date: datetime.date):
    return merge_with_date_range(data, generate_date_range(start_date, end_date))


def main(data_file, lookback, value_column, show_model):
    # Read the data
    df = pd.read_csv(data_file)
    df.rename(columns={value_column: 'value'}, inplace=True)

    # Convert date column to date format
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['date'] = pd.DatetimeIndex(df['date']).date

    # Merge the data with the date range to remove gaps
    df = fill_gaps_with_zeros(
        df, df['date'].min(), df['date'].max())

    # Get yesterday's date and subtract 12 months
    yesterday = (datetime.now() - timedelta(days=1)).date()
    lookback_days = (yesterday - timedelta(days=lookback))

    def get_filtered_data(data: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filters the given DataFrame using the given boolean mask, performs STL decomposition on the filtered data, and returns the filtered data, anomalies, and filtered trend.

        Parameters:
        data (pd.DataFrame): DataFrame containing time series data.
        data_filter (List[bool]): Boolean mask used to filter the data.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the filtered data, anomalies, and filtered trend.
        """
        # Create a boolean mask to filter the data
        data_filter = (data['date'] >
                       start_date) & (data['date'] <= end_date)

        # Filter the data and trend
        filtered_data = fill_gaps_with_zeros(
            data[data_filter], start_date, end_date)

        # Perform STL decomposition on filtered data
        stl = STL(
            filtered_data['value'].values, period=7).fit()

        filtered_trend = pd.DataFrame(
            {'date': filtered_data['date'], 'trend': stl.trend})

        # Calculate the residuals and the threshold for anomalies
        filtered_resid = stl.resid
        threshold = 1.5 * np.std(filtered_resid)

        # Detect anomalies
        anomalies = filtered_data[np.abs(filtered_resid) > threshold]

        if show_model:
            # Get the y-axis limits of the original data
            y_max = filtered_data['value'].max()

            # Plot the original data and the decomposed components
            fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

            ax[0].plot(filtered_data['date'], filtered_data['value'])
            ax[0].set_ylabel('Original Data')
            ax[0].scatter(anomalies['date'], anomalies['value'],
                          c='red', label='Current Year Anomalies')
            ax[0].set_ylim([0, y_max])

            ax[1].plot(filtered_data['date'], stl.trend)
            ax[1].set_ylabel('Trend')
            ax[1].set_ylim([0, y_max])

            ax[2].plot(filtered_data['date'], stl.seasonal)
            ax[2].set_ylabel('Seasonal')
            ax[2].set_ylim([-y_max/2, y_max/2])

            ax[3].plot(filtered_data['date'], stl.resid)
            ax[3].set_ylabel('Residual')
            ax[3].set_ylim([-y_max/2, y_max/2])

            plt.show()

        return filtered_data, anomalies, filtered_trend

    # Get the data for the current year
    current_year, anomalies_current, current_year_trend = get_filtered_data(
        df, lookback_days, yesterday)

    # Get the data for the previous year
    previous_year, anomalies_previous, previous_year_trend = get_filtered_data(df, lookback_days - timedelta(
        days=365), yesterday - timedelta(days=365))
    anomalies_previous = anomalies_previous.copy()
    anomalies_previous.loc[:, 'date'] = anomalies_previous['date'] + \
        pd.DateOffset(days=365)

    # Calculate the absolute difference between the two years
    diff = current_year['value'].values - \
        previous_year['value'].values
    diff_data = pd.DataFrame(
        {'date': current_year['date'], 'difference': diff})
    diff_data['yoy_diff'] = diff_data['difference'].rolling(
        window=7).mean()

    # Plot the observed data, trend, and seasonal components for the current and previous years
    fig, ax = plt.subplots(1, 1, figsize=(30, 10))

    # Customize the x-axis ticks
    dates = pd.date_range(start=lookback_days.isoformat(),
                          end=yesterday.isoformat(), freq='D')
    ax.set_xticks(dates[::7])
    ax.tick_params(axis='x', rotation=45)

    # set the date format for the x-axis
    date_fmt = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_formatter(date_fmt)

    # set the x-axis major tick locator to WeekdayLocator with MONDAY parameter
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO))

    ax.plot(current_year['date'],
            current_year['value'], label='Current Year')
    ax.plot(current_year_trend['date'],
            current_year_trend['trend'], linestyle='--', label='Current Year Trend')
    # Note: we use the current year date as the x index so it lines up as a YoY trend
    ax.plot(current_year['date'],
            previous_year['value'], label='Previous Year', alpha=0.4)
    ax.plot(current_year_trend['date'],
            previous_year_trend['trend'], linestyle='--', color="black", label='Previous Year Trend', alpha=0.4)
    ax.bar(diff_data['date'], diff_data['yoy_diff'],
           label="7 Day Rolling Avg YoY Diff", alpha=0.4)
    ax.scatter(anomalies_current['date'], anomalies_current['value'],
               c='red', label='Current Year Anomalies')
    ax.scatter(anomalies_previous['date'], anomalies_previous['value'],
               c='red', alpha=0.2, label='Previous Year Anomalies')
    ax.legend(loc='best')
    ax.set_title(
        'Observed Data, Trend, and YoY Comparison')

    # Set the minimum value for the y-axis
    min_max_y_value = 100

    # Set the new y-axis limits with the desired minimum maximum value
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max(current_ylim[1], min_max_y_value))

    plt.show()


# Set up argparse to parse command line arguments
parser = argparse.ArgumentParser(
    description='Process time series data and plot trends')
parser.add_argument('data_file', type=str,
                    help='name of CSV file with the time series data')
parser.add_argument('-d', '--lookback', type=int, default=90,
                    help='number of days of lookback for trend analysis (default: 90)')
parser.add_argument('-m', '--show-model', action='store_true',
                    help='Show the STL model')
parser.add_argument('-v', '--value-column', type=str, default='value',
                    help='Specify the column name for the value to analyze (default: "value"))')
args = parser.parse_args()
main(args.data_file, args.lookback, args.value_column, args.show_model)
