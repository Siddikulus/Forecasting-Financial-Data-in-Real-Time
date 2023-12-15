import pandas as pd
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse
import matplotlib.dates as mdates


def auto_arima_model(df, output_parameter):
    """
    Perform AutoARIMA modeling on the given DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): The input DataFrame containing time series data.
    - output_parameter (str): The name of the column to be predicted.
    - exogenous_variables (list): List of column names for exogenous variables.

    Returns:
    - model: The trained AutoARIMA model.
    - test_rmse: Root Mean Squared Error on the test set.
    """

    # Extract the relevant columns from the DataFrame
    # df = data_frame[[output_parameter]].copy()

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Quarter'].map({'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}), format='%Y%m')
    df.set_index('Date', inplace=True)

    # ----------------------------First Model Instance for Accuracy metrics---------------------------------------------------------------
    # Split the data into training and testing sets
    exog_train = df[df.index <= '2022-04-01']
    exog_test = df[df.index > '2022-04-01']
    exog_test = exog_test[exog_test.index <= '2023-04-01']

    # Perform AutoARIMA model selection
    model = auto_arima(start_p=1, start_q=1, test='adf',
                       y=exog_train[output_parameter],
                       exogenous=exog_train,
                       seasonal=True,
                       suppress_warnings=True,
                       stepwise=True,
                       m=4,  # Assuming quarterly data with a seasonality of 4
                       max_p=5,  # Maximum number of autoregressive terms
                       max_q=5,  # Maximum number of moving average terms
                       d=1,
                       max_order=None  # Maximum order of differencing
                       )

    # Make predictions on the test set
    forecast, conf_int = model.predict(n_periods=len(exog_test), exogenous=exog_test, return_conf_int=True)
    # Calculate Root Mean Squared Error on the test set
    test_rmse = sqrt(mean_squared_error(exog_test[output_parameter], forecast))
    # predicted_values = pd.Series(forecast, index=exog_test.index, name='Predicted')
    predicted_values = pd.DataFrame({
        str(output_parameter): list(forecast)
    })
    predicted_values.index = exog_test.index
    return predicted_values, test_rmse