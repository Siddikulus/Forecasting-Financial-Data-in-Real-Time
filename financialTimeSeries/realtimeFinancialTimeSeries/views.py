from django.shortcuts import render
import pandas as pd
from .models import SampleData, TargetVariable, CorrelationMatrix
from sqlalchemy import create_engine
from django.conf import settings
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from realtimeFinancialTimeSeries.preprocessing import standardize
from realtimeFinancialTimeSeries.preprocessing import PCA
from realtimeFinancialTimeSeries.preprocessing import drophighlycorrelated
from realtimeFinancialTimeSeries.forecastingmodels import arimamodel
from realtimeFinancialTimeSeries.forecastingmodels import holtwinters


current_working_directory = os.getcwd()


db_connection_url = "postgresql://{}:{}@{}:{}/{}".format(
    settings.DATABASES['default']['USER'],
    settings.DATABASES['default']['PASSWORD'],
    settings.DATABASES['default']['HOST'],
    settings.DATABASES['default']['PORT'],
    settings.DATABASES['default']['NAME'],
)

engine = create_engine(db_connection_url)

def correlation_graph(df, target_variable, exog_var):
    plt.figure(figsize=(7, 6))
    plt.plot(df.index, df[target_variable], color = 'red', linewidth=2.3, label = str(target_variable))
    plt.plot(df.index, df[exog_var], color='blue', linewidth=2.3, label = str(exog_var))
    plt.legend(loc="upper left")
    plt.xticks(rotation=45, ha='right')
    plt.xticks(df.index[::4])
    plt.yticks([])

    loc = os.path.join('static','image',target_variable,exog_var+'.png')
    path = os.path.join(current_working_directory,'realtimeFinancialTimeSeries','static','image',target_variable)
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(os.path.join(current_working_directory,'realtimeFinancialTimeSeries',loc))

def index(request):
    return render(request, 'index.html')

def select_target_variable(request):
    if 'excelfileupload' in request.POST:
        filepath = request.FILES['excelfile']

        df = pd.read_excel(filepath, sheet_name = 'Raw_Data')
        df_corr = df.drop(['Quarter', 'Year'], axis = 1)


        df = pd.melt(df, id_vars = ['Quarter', 'Year'], var_name = 'feature', value_name = 'value')
        df['id'] = df.index
        df = df[['id', 'Quarter', 'Year', 'feature', 'value']]
        df.to_sql(SampleData._meta.db_table, engine, if_exists='replace', chunksize=10000, index = False)

        df_target = pd.read_excel(filepath, sheet_name = 'Target_Variable')
        df_target['id'] = df_target.index
        df_target = df_target[['id', 'Target']]
        df_target.to_sql(TargetVariable._meta.db_table, engine, if_exists = 'replace', index = False)

        target_variables = list(df_target['Target'])
        request.session['all_target_variables'] = target_variables

        df_corr = df_corr.corr().reset_index()
        df_corr = pd.melt(df_corr, id_vars = ['index'], var_name = 'correlation_to', value_name = 'correlation_coefficient')
        df_corr = df_corr.rename(columns={'index': 'correlation_from'})
        df_corr['id'] = df_corr.index
        df_corr = df_corr[['id', 'correlation_from', 'correlation_to', 'correlation_coefficient']]
        df_corr.to_sql(CorrelationMatrix._meta.db_table, engine, if_exists='replace', chunksize=10000, index = False)

        return render(request, 'selecttarget.html', {'target_variables': target_variables})

def eda(request):
    target_variable = (request.POST.get('select_target', False)).strip()
    request.session['target'] = target_variable

    df_target = pd.DataFrame(list(SampleData.objects.all().values()))
    df_target['Time_Period'] = df_target['Quarter'].replace('Q1', '01-03').replace('Q2', '01-06').replace('Q3', '01-09').replace('Q4', '01-12')
    df_target['Date'] = df_target.apply(lambda x: datetime.strptime(x['Time_Period'] + '-' + str(x['Year']), '%d-%m-%Y'), axis = 1)

    df_data = df_target.copy()

    df_target = df_target[df_target['feature'] == target_variable]
    df_target.index = df_target['Date']
    df_target = df_target[df_target.index <= '01-06-2023']
    df_decompose = df_target.drop(['Quarter', 'Year', 'id', 'Date', 'feature', 'Time_Period'], axis = 1)

    #Decomposing target varibale for trendline
    result = seasonal_decompose(df_decompose, model='additive')
    trend = result.trend
    seasonality = result.seasonal
    residual = result.resid

    #Plotting EDA plots
    fig, axes = plt.subplots(2, 2, figsize=(13,8), sharex=False, layout="constrained")
    axes[0,0].plot(df_decompose.index, df_decompose['value'], color = 'red')
    axes[0,0].set_title(r'Original Trend', fontsize=13)
    axes[0,0].set_xlabel(r'Year')
    axes[0,0].set_ylabel(target_variable)
    axes[0,0].grid()

    axes[0,1].plot(df_decompose.index, trend)
    axes[0,1].set_title(r'Trend Line', fontsize=13)
    axes[0,1].set_xlabel(r'Year')
    axes[0,1].set_ylabel(target_variable)
    axes[0,1].grid()

    axes[1,0].plot(df_decompose.index, seasonality)
    axes[1,0].set_title(r'Seasonality', fontsize=13)
    axes[1,0].set_xlabel(r'Year')
    axes[1,0].grid()

    axes[1,1].plot(df_decompose.index, residual)
    axes[1,1].set_title(r'Residual', fontsize=13)
    axes[1,1].set_xlabel(r'Year')
    axes[1,1].grid()

    loc = os.path.join('static','image','seasonal_decompose.png')
    plt.savefig(os.path.join(current_working_directory,'realtimeFinancialTimeSeries',loc))
    plt.clf()

    exog_vars = list(set(df_data['feature'].unique()) - set(target_variable))
    df_data = df_data.pivot_table(columns = 'feature', values = 'value', index = ['Quarter',  'Year'])
    df_data = df_data.reset_index().sort_values(['Year', 'Quarter'])
    df_data.index = df_data.apply(lambda x: x['Quarter']+' '+str(x['Year']), axis = 1)
    df_data = df_data.drop(['Quarter', 'Year'], axis = 1)

    scaler = MinMaxScaler()
    df_data[df_data.columns] = scaler.fit_transform(df_data[df_data.columns])

    for var in exog_vars:
        correlation_graph(df_data, target_variable, var)

    return render(request, 'eda.html', {"target": target_variable, "imagepath":loc})

def correlation(request):
    target_variable = request.session['target']
    all_target_variables = request.session['all_target_variables']
    df_corr = pd.DataFrame(list(CorrelationMatrix.objects.all().values()))
    df_corr = df_corr[df_corr['correlation_from'] == target_variable]
    df_corr = df_corr[df_corr['correlation_coefficient'] != 1]
    df_corr = df_corr[~df_corr['correlation_to'].isin(all_target_variables)]
    df_corr = df_corr.drop(['id', 'correlation_from'], axis = 1)
    df_corr['correlation_coefficient'] = df_corr['correlation_coefficient'].apply(lambda x: round(x, 3))
    df_corr['absolute_coefficients'] = df_corr['correlation_coefficient'].apply(lambda x: abs(x), 2)


    columns = list(df_corr.columns)
    exogenous_variables = list(df_corr['correlation_to'])
    coeff = list(df_corr['correlation_coefficient'])

    imagepath = os.path.join('static','image',target_variable)+"\\"

    return render(request, 'correlation.html', {"correlation_matrix":df_corr,"columns":columns, "exogenous_variables":exogenous_variables,"coeff":coeff,"target_variable": target_variable, "imagepath":imagepath})

def modelselection(request):
    selected_features = request.POST.getlist('exogvars[]', False)
    request.session['selected_exogenous_variables'] = selected_features

    df_data = pd.DataFrame(list(SampleData.objects.all().values()))
    df_data['Time_Period'] = df_data['Quarter'].replace('Q1', '01-03').replace('Q2', '01-06').replace('Q3', '01-09').replace('Q4', '01-12')
    df_data['Date'] = df_data.apply(lambda x: datetime.strptime(x['Time_Period'] + '-' + str(x['Year']), '%d-%m-%Y'), axis = 1)
    df_data = df_data.pivot_table(columns='feature', values='value', index=['Quarter', 'Year'])
    df_data = df_data.reset_index().sort_values(['Year', 'Quarter'])

    df_multicollinearity = df_data[selected_features]


    #Heatmap
    corr = df_multicollinearity.corr()

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        corr,
        cmap=sns.cubehelix_palette(as_cmap=True),
        square=True, annot = True
    )
    ax.set_ylabel('')
    ax.set_xlabel('')
    loc = os.path.join('static','image','heatmap.png')
    plt.savefig(os.path.join(current_working_directory,'realtimeFinancialTimeSeries',loc), bbox_inches='tight')
    plt.clf()

    return render(request, 'modelselection.html',{"exogvars": selected_features, "imagepath": loc})

def evaluation(request):
    selected_models = request.POST.getlist('modelselection[]', [])
    selected_preprocessing_model = request.POST.getlist('preprocessing[]', [])
    target_variable = request.session['target']
    selected_exog_vars = request.session['selected_exogenous_variables']


    #Getting Input Data
    df_data = pd.DataFrame(list(SampleData.objects.all().values()))

    df_data = df_data.pivot_table(columns='feature', values='value', index=['Quarter', 'Year'])
    df_data = df_data.reset_index().sort_values(['Year', 'Quarter'])
    df_data['Time_Period'] = df_data['Quarter'].replace('Q1', '01-03').replace('Q2', '01-06').replace('Q3',
                                                                                                      '01-09').replace(
        'Q4', '01-12')
    df_data['Date'] = df_data.apply(lambda x: datetime.strptime(x['Time_Period'] + '-' + str(x['Year']), '%d-%m-%Y'),
                                    axis=1)
    df_data.index = df_data['Date']

    preprocessed_df = df_data.copy()
    preprocessed_df = preprocessed_df.drop(['Date', 'Quarter', 'Year', 'Time_Period', target_variable], axis=1)
    preprocessed_df = preprocessed_df[selected_exog_vars]

    #Running Preprocessing Algorithms
    if 'HighCorrDrop' in selected_preprocessing_model:
        preprocessed_df = drophighlycorrelated.drophighcorr(preprocessed_df)

    if 'standardize' in selected_preprocessing_model:
        preprocessed_df = standardize.standard(preprocessed_df)

    if 'PCA' in selected_preprocessing_model:
        preprocessed_df = PCA.principalcomponentanalysis(preprocessed_df, target_variable)

    if target_variable not in preprocessed_df.columns:
        preprocessed_df[target_variable] = df_data[target_variable]

    #Running Forecasting Models
    exogenous_variables = list(set(preprocessed_df.columns) - set([target_variable, 'Date', 'Quarter', 'Year', 'Time_Period']))
    model_results = pd.DataFrame(index = preprocessed_df.index[-4:])
    evaluation_matrix = pd.DataFrame()

    if 'ARIMA' in selected_models:
        arima_predicted_values, arima_rmse, arima_mae, arima_mape = arimamodel.auto_arima_model(preprocessed_df, target_variable, exogenous_variables)
        model_results['SARIMA Predictions'] = arima_predicted_values
        evaluation_matrix = pd.concat([evaluation_matrix,pd.DataFrame({'Model': ['SARIMA'], 'RMSE': [arima_rmse], 'MAE': [arima_mae], 'MAPE': [arima_mape]})])

    if 'Holt' in selected_models:
        holt_predicted_values, holt_rmse, holt_mae, holt_mape = holtwinters.holtwintersmodel(preprocessed_df, target_variable, exogenous_variables)
        model_results['Holt-Winters Predictions'] = holt_predicted_values
        evaluation_matrix = pd.concat([evaluation_matrix,pd.DataFrame({'Model': ['Holt Winters'], 'RMSE': [holt_rmse], 'MAE': [holt_mae], 'MAPE': [holt_mape]})])


    #Changing index to 'Quarter-Year'
    model_results = model_results.reset_index()
    model_results['NewDate'] = model_results.apply(lambda x: datetime.strftime(x['Date'], '%d-%m-%Y').replace('01-03', 'Q1').replace('01-06', 'Q2').replace('01-09','Q3').replace('01-12', 'Q4'), axis = 1)
    model_results = model_results.set_index('NewDate')

    df_data['NewDate'] = df_data.apply(
        lambda x: datetime.strftime(x['Date'], '%d-%m-%Y').replace('01-03', 'Q1').replace('01-06', 'Q2').replace(
            '01-09', 'Q3').replace('01-12', 'Q4'), axis=1)
    df_data = df_data.set_index('NewDate')

    #Plotting Forecasts
    plt.figure(figsize=(13, 8))
    plt.plot(df_data.index, df_data[target_variable], color = 'blue', linewidth=2.3, label = str(target_variable))
    plt.plot(model_results.index, model_results['SARIMA Predictions'], color='green', linewidth=2.3, label = 'SARIMA')
    plt.plot(model_results.index, model_results['Holt-Winters Predictions'], color='purple', linewidth=2.3, label = 'Holt-Winters')
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel('Time Period')
    plt.ylabel(target_variable)
    plt.xticks(rotation=45, ha='right')
    plt.xticks(df_data.index[::4])

    loc = os.path.join('static','image','model_evaluation.png')
    plt.savefig(os.path.join(current_working_directory,'realtimeFinancialTimeSeries',loc), bbox_inches='tight')

    #Evaluation Tables
    model_results = model_results.drop(['Date'], axis = 1)
    model_results.index.name = None
    evaluation_matrix.index.name = None

    return render(request, 'evaluation.html',{"imagepath": loc, "exogenous_variables": ', '.join(selected_exog_vars), 'predictedvals': model_results.to_html(classes = "table table-sm table-hover table-dark"), 'evaluation_matrix': evaluation_matrix.to_html(classes = "table table-sm table-hover table-dark", index = False)})



