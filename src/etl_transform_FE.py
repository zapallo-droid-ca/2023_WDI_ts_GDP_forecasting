
import pandas as pd


def dummies_var(df, time_var):
    df['finantial_crisis'] = (df[time_var] >= 2007) & (df[time_var] <= 2008)
    df['pandemic'] = (df[time_var] >= 2020) & (df[time_var] <= 2023)
    
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