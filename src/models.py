
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
from src import model_pipeline as mpln

##-- Reading Data:
wd = 'C:/Users/jrab9/OneDrive/08.Github/2023_WDI_ts_GDP_forecasting/'
    
df = pd.read_csv(wd + '/data/final/ft_tsd.csv.gz')
df = df.drop(columns = ['level_lag_0', 'seasonal_lag_0', 'residual_lag_0', 'trend_lag_0', 'level_lag_1', 'seasonal_lag_1',
                        'residual_lag_1', 'trend_lag_1', 'level_lag_2', 'seasonal_lag_2', 'residual_lag_2', 'trend_lag_2'])
df = df[['time','country_iso3','level']]

with open(wd + '/data/final/relevant_countries.txt', 'r') as f:
    relevant_countries = f.readlines()
f.close()

relevant_countries = [x.replace('\n','') for x in relevant_countries]
  
countries = pd.read_csv(wd + '/data/final/dim_country.csv.gz')[['iso3','name']].drop_duplicates().reset_index(drop = True)
countries = countries[countries['iso3'].isin(df['country_iso3'].unique())].reset_index(drop = True)
relevant_countries = countries[countries['iso3'].isin(relevant_countries)].reset_index(drop = True)

countries = list(zip(countries.iloc[:,0],countries.iloc[:,1]))
relevant_countries = list(zip(relevant_countries.iloc[:,0],relevant_countries.iloc[:,1]))

extra_relevants = [('AUS','Australia'),('SAU','Saudi Arabia'),('IDN','Indonesia')]
for x in extra_relevants:
    relevant_countries.append(x)

##-- Preprocessing:
cols_exclude_preprocess = ['time','country_iso3']

#PARAMETERS: Datasets
category_var = 'country_iso3'
time_var = 'time'
target_var = 'level'
train_size = 0.7

#PARAMETERS: Autoregressive Components (ARIMA)
lags = 10
threshold = 0.2
max_adf_iters = 3

#PARAMETERS HOWI
iterHowi = np.round(np.arange(0.05, 1.0, 0.05),4) #Range up to 1.05 to evaluate simple and doble exponential smoothing, step of 0.1 is not exhaustive but less cost
typeHowi = ['add']

## Modelos
models_summary = []
models_results = []

predict_index = list(range(max(df.time)+1,2031))

##--ARIMA
models_summary, models_results = mpln.arima_pipe(df = df, countries = relevant_countries, prep_scaler = True, prep_diff = True, cols_exclude_preprocess = cols_exclude_preprocess, 
                                                 time_var = time_var, target_var = target_var, lags = lags, threshold = threshold, max_adf_iters = max_adf_iters, 
                                                 models_summary = models_summary, models_results = models_results, category_var = category_var, train_size = train_size, 
                                                 predict_index = predict_index, wd = wd, scaler_minmax = True)

#PARAMETERS: Preprocessing
##--HOLT-WINTERS
models_summary, models_results = mpln.howi_pipe(df = df, countries = relevant_countries, prep_scaler = True, prep_diff = False, cols_exclude_preprocess = cols_exclude_preprocess, 
                                                 time_var = time_var, target_var = target_var, lags = lags, threshold = threshold, max_adf_iters = max_adf_iters, iterHowi = iterHowi, 
                                                 typeHowi = typeHowi, models_summary = models_summary, models_results = models_results, category_var = category_var, train_size = train_size, 
                                                 predict_index = predict_index, wd = wd, scaler_minmax = True)

