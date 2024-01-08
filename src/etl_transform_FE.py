
import pandas as pd
import numpy as np


def dummies_var(df, time_var):
    df['global_crisis'] = df[time_var].isin([2014,2015,2018,2020,2022])
    
    return df


def lags_creation_tsd(data, category, index_variable, lags_var, category_inter, number_of_lags, shift_value):
    
    category_inter = data[category].unique()
    
    df_temp = pd.DataFrame()
    
    created_vars = set()
    
    for i in category_inter:
        df_temp_iter = data[data[category] == i].sort_values(index_variable,ascending = True)  
    
        for i in range(0,number_of_lags):
            for j in lags_var:
                df_temp_iter[f'{j}_lag_{i}'] = df_temp_iter[j].shift(periods = (shift_value + i))
                
                created_vars.update([f'{j}_lag_{i}'])
                
        df_temp_iter = df_temp_iter.dropna(subset = list(created_vars)).reset_index(drop = True)
                
        df_temp = pd.concat([df_temp,df_temp_iter])    
    
    df_temp.reset_index(drop = True, inplace = True)
    
    return df_temp


def rank_creation(data, var_target, index_variable):
    
    years = data[index_variable].unique()
    
    dt_temp = pd.DataFrame()

    for i_year in years:           
        
        df_iter = data[data[index_variable] == i_year].sort_values(var_target, ascending = False).reset_index(drop = True)
        df_iter[f'{var_target}_rank'] = df_iter[var_target].rank(method = 'dense', ascending = False)
        
        dt_temp = pd.concat([dt_temp,df_iter])
        
    return dt_temp.sort_values([index_variable,f'{var_target}_rank'], ascending = True).reset_index(drop = True)


def diff_creation_tsd(data, category, index_variable, var_target):
    
    category_inter = data[category].unique()
    
    df_temp = pd.DataFrame()   
    
    for i in category_inter:
        df_temp_iter = data[data[category] == i].sort_values(index_variable,ascending = True)      
                
        df_temp_iter[f'{var_target}_diff'] = df_temp_iter[var_target].diff()
            
        df_temp = pd.concat([df_temp,df_temp_iter])    
    
    df_temp.reset_index(drop = True, inplace = True)
    
    return df_temp


def pct_growth_tsd(data, category, index_variable, var_target):
    
    category_inter = data[category].unique()
    
    df_temp = pd.DataFrame()   
    
    for i in category_inter:
        df_temp_iter = data[data[category] == i].sort_values(index_variable,ascending = True)      
                
        df_temp_iter[f'{var_target}_growth'] = np.round((df_temp_iter[var_target] - df_temp_iter[var_target].shift(1)) / df_temp_iter[var_target].shift(1),2) * 100
        df_temp = pd.concat([df_temp,df_temp_iter])    
    
    df_temp.reset_index(drop = True, inplace = True)
    
    return df_temp