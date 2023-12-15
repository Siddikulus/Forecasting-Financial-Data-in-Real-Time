import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def holtwintersmodel(df, target_column, exogenous_variables):
    #
    # df['date'] = df['Year'].astype(str) + df['Quarter']
    # quarter_mapping = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
    # df['date'] = df['date'].replace(quarter_mapping, regex=True)
    # df['date'] = pd.to_datetime(df['date'], format='%Y%m')
    #
    # # Drop the original 'Quarter' and 'Year' columns
    # df = df.drop(['Quarter', 'Year'], axis=1)
    #
    # # Set the 'date' column as the index
    # df = df.set_index('date')
    # Specify the target column

    # ----------------------------First Model Instance for Accuracy metrics---------------------------------------------------------------
    # Split the data into training and testing sets
    train = df[df.index <= '2022-06-01']
    test = df[df.index > '2022-06-01']
    test = test[test.index <= '2023-06-01']

    # Smoothing Parameters
    alpha = 0.9  # Smoothing parameter for the level (trend)
    beta = 0.9  # Smoothing parameter for the trend
    gamma = 0.3  # Smoothing parameter for the seasonality

    # Perform Holt-Winters Time Series Forecast on the training set
    model = ExponentialSmoothing(train[target_column], seasonal='add', seasonal_periods=4)
    fitted_model = model.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)

    # Forecast for the test set
    forecast = fitted_model.forecast(steps=len(test))

    # Calculate Root Mean Squared Error on the test set
    test_rmse = sqrt(mean_squared_error(test[target_column], forecast))
    test_mae = mean_absolute_error(test[target_column], forecast)
    test_mape = mean_absolute_percentage_error(test[target_column], forecast) * 100


    # ----------------------------Second Model Instance for Accuracy metrics---------------------------------------------------------------
    # Split the data into training and testing sets
    train = df[df.index <= '2023-06-01']
    test = df[df.index > '2023-06-01']


    # Smoothing Parameters
    alpha = 0.9  # Smoothing parameter for the level (trend)
    beta = 0.9  # Smoothing parameter for the trend
    gamma = 0.3  # Smoothing parameter for the seasonality

    # Perform Holt-Winters Time Series Forecast on the training set
    model = ExponentialSmoothing(train[target_column], seasonal='add', seasonal_periods=4)
    fitted_model = model.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)

    # Forecast for the test set
    forecast = fitted_model.forecast(steps=len(test))
    predicted_values = pd.Series(forecast, index=test.index, name='Predicted')
    return predicted_values, test_rmse, test_mae, test_mape