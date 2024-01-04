### Project ETL

## Libraries
import pandas as pd
import numpy as np
import os
import requests
import json
import sqlite3

### Custom Libraries

from src import etl_extract, etl_qa, etl_transform, etl_transform_TS, etl_transform_FE

## Work Directory
wd = os.getcwd().replace('\src','\\')

## General configuration
number_of_lags = 3
range_min = 1990
range_max = 2022

# Variables in lag creation:
lags_level = True
lags_trend = True
lags_seasonal = True
lags_residual = True

#Number of differenciations (periods of pd.dataframe.shift())
shift_value = 1


#-- ETL: Extract
data, dim_country = etl_extract.wdi_extract(wd, range_min = range_min, range_max = range_max) 



##-- ETL: Transform

##--- 1: base dataframes
ft_wdi, ft_nas = etl_extract.base_data(wd = wd, nas_df = True)

#QA
etl_qa.time_range_complete(df = ft_wdi, category_var = 'economy', time_var = 'time')

#Filter: Main variable / target completition
ft_wdi['target_completition'] = ft_wdi['GDP_USD'] == 0 #considering that a country GDP should be > 0

#Extracting lost subjects
etl_log = pd.Series(ft_wdi[ft_wdi['target_completition']]['economy'].unique(), name = 'economy_filtered')
etl_log.to_csv(wd + '/data/etl/observation_filteres.csv', index = False)

ft_wdi = ft_wdi[ft_wdi['economy'].isin(etl_log) == False].sort_values(['economy','time'], ascending = True).reset_index(drop = True).drop(columns = 'target_completition')
ft_nas = ft_nas[ft_nas['economy'].isin(etl_log) == False].sort_values(['economy','time'], ascending = True).reset_index(drop = True)
dim_country = dim_country[dim_country['Country ISO3'].isin(etl_log) == False].sort_values('Country ISO3', ascending = True).reset_index(drop = True)

##--- 2: TSD
#Getting Data
ft_tsd = ft_wdi[['economy', 'time', 'GDP_USD']].copy().sort_values(['economy','time'], ascending = True)
ft_tsd.set_index(pd.to_datetime(ft_tsd['time'], format = '%Y',), inplace = True)


ft_tsd = etl_transform_TS.tsDecomposition(data = ft_tsd, index_frequency = 'YS', period = number_of_lags, #Taking a seasonal period equivalent to number_of_lags
                                          target = 'GDP_USD', category = 'economy', sub_category = None, 
                                          sdModel = 'additive', two_sided = False)

ft_tsd['date'] = ft_tsd['date'].dt.year
ft_tsd.columns = ft_tsd.columns.str.replace('date','time')

#Join TSD with WDI base
ft_tsd_temp = ft_tsd[['time','economy','trend','seasonal','residual']].copy()
ft_tsd_temp.columns = ['time','economy','GDP_USD_trend','GDP_USD_seasonal','GDP_USD_residual']

ft_wdi = ft_wdi.merge(ft_tsd_temp, how = 'left', left_on = ['time','economy'], right_on = ['time','economy'])
del(ft_tsd_temp)

#QA
etl_qa.time_range_complete(df = ft_tsd, category_var = 'economy', time_var = 'time')
ft_tsd['level'].sum() == ft_wdi['GDP_USD'].sum()
ft_tsd.shape[0] == ft_wdi.shape[0]


##-- Outliers Detection
#HAMPEL FILTER FOR TS OUTLIERS  
ft_tsd, df_eda = etl_transform_TS.hampel_filter(data = ft_tsd, category_var = 'economy', target = ['level','residual'], time_var = 'time', 
                                                windows_size = number_of_lags, n_sigmas = 3)

##--- 3: Feature Engineering
##-- Dummies variables
ft_wdi = etl_transform_FE.dummies_var(df = ft_wdi, time_var = 'time')
ft_tsd = etl_transform_FE.dummies_var(df = ft_tsd, time_var = 'time')
df_eda = etl_transform_FE.dummies_var(df = df_eda, time_var = 'time')

##-- GDP Lags
# Creating variables iterator
lags_var = set()

if lags_level: 
    lags_var.update(['level'])
if lags_trend:
    lags_var.update(['trend'])
if lags_seasonal:
    lags_var.update(['seasonal'])
if lags_residual:
    lags_var.update(['residual']) 

# Creating Lags
ft_tsd = etl_transform_FE.lags_creation_tsd(data = ft_tsd, category = 'economy', index_variable = 'time', lags_var = lags_var,
                                            category_inter = lags_var, number_of_lags = number_of_lags, shift_value = shift_value)

df_eda = etl_transform_FE.lags_creation_tsd(data = df_eda, category = 'economy', index_variable = 'time', lags_var = lags_var,
                                            category_inter = lags_var, number_of_lags = number_of_lags, shift_value = shift_value)


##--- 4: Data Load
#Eportar a csv y sqlite ft_tsd, df_eda, ft_wdi, ft_nas, dim_country
# .csv files
df_eda.to_csv(wd + '/data/processed/df_eda.csv.gz', index = False)
ft_tsd.to_csv(wd + '/data/processed/ft_tsd.csv.gz', index = False)
ft_wdi.to_csv(wd + '/data/processed/ft_wdi.csv.gz', index = False)
ft_nas.to_csv(wd + '/data/processed/ft_nas.csv.gz', index = False)
dim_country.to_csv(wd + '/data/final/dim_country.csv.gz', index = False)














    
    


