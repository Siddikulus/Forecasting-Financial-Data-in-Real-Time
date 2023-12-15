import pandas as pd
import numpy as np
from Models import arima_forecasting

input_df = pd.read_excel('Input_Data/Walmart_Revenue.xlsx', sheet_name = 'Raw_Data')
target_df = pd.read_excel('Input_Data/Walmart_Revenue.xlsx', sheet_name = 'Target_Variable')

target_vars = list(target_df['Target'])

input_df = input_df.drop(target_vars, axis = 1)

final_predicted_df = pd.DataFrame()
for exog_var in list(set(input_df.columns)-set(['Quarter', 'Year'])):
    temp_df = input_df[['Quarter', 'Year', exog_var]]
    predicted, acc = arima_forecasting.auto_arima_model(temp_df, exog_var)
    final_predicted_df = pd.concat([final_predicted_df, predicted], axis = 1)


final_predicted_df.reset_index().to_excel(r'Output_Data/Walmart_Revenue.xlsx')
