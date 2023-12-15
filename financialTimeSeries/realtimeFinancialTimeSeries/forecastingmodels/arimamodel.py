import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

def auto_arima_model(data_frame, output_parameter, exogenous_variables):
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
    df = data_frame[[output_parameter] + exogenous_variables].copy()

    # ----------------------------First Model Instance for Accuracy metrics---------------------------------------------------------------
    # Split the data into training and testing sets
    exog_train = df[df.index <= '2022-06-01']
    exog_test = df[df.index > '2022-06-01']
    exog_test = exog_test[exog_test.index <= '2023-06-01']


    # Perform AutoARIMA model selection
    model = auto_arima(y=exog_train[output_parameter],
                       X=exog_train[exogenous_variables],
                       seasonal=True,
                       suppress_warnings=True,
                       stepwise=True,
                       m=4,  # Assuming quarterly data with a seasonality of 4
                       max_p=5,  # Maximum number of autoregressive terms
                       max_q=5,  # Maximum number of moving average terms
                       d=1,
                       max_order=None  # Maximum order of differencing
                       )
    print(model.summary())
    model.fit(exog_train[output_parameter], X=exog_train[exogenous_variables])
    # Make predictions on the test set
    forecast, conf_int = model.predict(n_periods=len(exog_test), X=exog_test[exogenous_variables], return_conf_int=True)

    # Calculate Root Mean Squared Error on the test set
    test_rmse = sqrt(mean_squared_error(exog_test[output_parameter], forecast))
    test_mae = mean_absolute_error(exog_test[output_parameter], forecast)
    test_mape = mean_absolute_percentage_error(exog_test[output_parameter], forecast) * 100


    # ----------------------------Second Model Instance for Prediction---------------------------------------------------------------
    # Split the data into training and testing sets
    exog_train = df[df.index <= '2023-06-01']
    exog_test = df[df.index > '2023-06-01']

    # Perform AutoARIMA model selection
    model = auto_arima(y=exog_train[output_parameter],
                       X=exog_train[exogenous_variables],
                       seasonal=True,
                       suppress_warnings=True,
                       stepwise=True,
                       m=4,  # Assuming quarterly data with a seasonality of 4
                       max_p=5,  # Maximum number of autoregressive terms
                       max_q=5,  # Maximum number of moving average terms
                       d=1,
                       max_order=None  # Maximum order of differencing
                       )
    print(model.summary())
    model.fit(exog_train[output_parameter], X=exog_train[exogenous_variables])
    # Make predictions on the test set
    forecast, conf_int = model.predict(n_periods=len(exog_test), X=exog_test[exogenous_variables], return_conf_int=True)
    predicted_values = pd.Series(forecast, index=exog_test.index, name='Predicted')
    return predicted_values, test_rmse, test_mae, test_mape