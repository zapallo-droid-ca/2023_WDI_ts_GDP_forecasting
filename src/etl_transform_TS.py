import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from fastdtw import fastdtw

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

##-- TIME SERIES DECOMPOSITION
def tsDecomposition(data, index_frequency, period, target, category, sub_category = None, sdModel = 'additive', two_sided = False):
    """
    The function should receive a pandas dataframe aggregated with the category and sub_category vars and with the timestamp in the index.

    Where:
        * index_frequency (str): parameter for .asfreq() -- 'YS: Annual Std', 'm', 'A: Annual'
        * period (int): parameter period of seasonal_decompose (YS: 12, Annual: 12)
        * sdModel (str): parameter for statsmodels seasonal_decompose -- 'additive' / 'multiplicative'
        * two_sided (bool): parameter for statsmodels seasonal_decompose
    """
    #Categories to be Iterated
    if sub_category == None:
        category_inter = data[category].unique()
    else:       
        category_inter = tuple(data[[category,sub_category]].drop_duplicates().itertuples(index = False))

    #Future Dataframes to be created after loops
    df_trend = pd.DataFrame()
    df_seasonal = pd.DataFrame()
    df_residual = pd.DataFrame()
    df_level = pd.DataFrame()

    #Time Series Decomposition (Scaled by each category)
    
    for iterator in category_inter:
        if sub_category == None:
            cat = iterator
            subcat = sub_category
            #-Filtering
            df_temp = data[data[category] == cat].sort_index(ascending = True)  

        else:      
            cat = iterator[0]
            subcat = iterator[1]

            #-Filtering
            df_temp = data[(data[category] == cat) & (data[sub_category] == subcat)].sort_index(ascending = True)

        #-Indexing
        df_temp = df_temp.asfreq(index_frequency)
        
        series_level = df_temp[target].copy()

        #-Time Series Decomposition
        sdDF = seasonal_decompose(df_temp[target], model = sdModel, period = period, two_sided = two_sided)
            
        temp_level = {'date': series_level.index,
                      'value': series_level.values,
                      category:cat,
                      sub_category:subcat,
                      'target':target,
                      'component':'level'}

        temp_trend = {'date': sdDF.trend.index,
                    'value': sdDF.trend.values,
                    category:cat,
                    sub_category:subcat,
                    'target':target,
                    'component':'trend'}

        temp_seasonal = {'date': sdDF.seasonal.index,
                        'value': sdDF.seasonal.values,
                        category:cat,
                        sub_category:subcat,
                        'target':target,
                        'component':'seasonal'}

        temp_resid = {'date': sdDF.resid.index,
                    'value': sdDF.resid.values,
                    category:cat,
                    sub_category:subcat,
                    'target':target,
                    'component':'residual'}

        df_trend = pd.concat([df_trend,pd.DataFrame(temp_trend)])
        df_seasonal = pd.concat([df_seasonal,pd.DataFrame(temp_seasonal)])
        df_residual = pd.concat([df_residual,pd.DataFrame(temp_resid)])
        df_level = pd.concat([df_level,pd.DataFrame(temp_level)])        

        del(temp_trend,temp_seasonal,temp_resid,temp_level, sdDF)  
        print(f'{target} by {cat} done for subcategory {subcat}', end = '\r') 

    df_timeSeriesDecomp = pd.concat([df_trend,df_seasonal,df_residual,df_level]).reset_index(drop = True)
    components_pivot = ['level','trend','seasonal','residual']   

    if sub_category == None:
        df_timeSeriesDecomp = df_timeSeriesDecomp[df_timeSeriesDecomp['component'].isin(components_pivot)].pivot_table(index = ['date',category], columns = 'component', values = 'value', aggfunc = 'sum').reset_index()
    else:
        df_timeSeriesDecomp = df_timeSeriesDecomp[df_timeSeriesDecomp['component'].isin(components_pivot)].pivot_table(index = ['date',category,sub_category], columns = 'component', values = 'value', aggfunc = 'sum').reset_index()
    df_timeSeriesDecomp.columns = df_timeSeriesDecomp.columns.rename('')
    df_timeSeriesDecomp.sort_values('date', ascending = True, inplace = True)
           
    print(f'process ended, df_timeSeriesDecomp: {df_timeSeriesDecomp.shape}')
    
    warning_label = 'component is 0, please check if the function is working well or if you have problem loading data or the parameters (for example check the period and index_frequency, should be consistent)'

    if df_trend.value.sum() == 0:
        component_label = 'trend'
        print(f'{component_label} {warning_label}')
    if df_seasonal.value.sum() == 0:
        component_label = 'seasonal'
        print(f'{component_label} {warning_label}')
    if df_residual.value.sum() == 0:
        component_label = 'residual'
        print(f'{component_label} {warning_label}')

    return df_timeSeriesDecomp


##-- HAMPEL FILTER FOR TS OUTLIERS
def hampel_filter(data, category_var, target, time_var, windows_size, n_sigmas = 3):
    """
    Apply the Hampel filter to a time series.
    
    Parameters:
    input_series (pd.Series): The input time series to filter.
    window_size (int): The size of the rolling window (odd integer).
    n_sigmas (int): The number of standard deviations to use as the threshold.
    
    Returns:
    pd.DataFrame: The filtered data
    """
    
    result = pd.DataFrame()   
    
    category_inter = data[category_var].unique()
    
    cols_to_drop = set()
    
    for i in category_inter:
        df_temp = data[data[category_var] == i].sort_values(time_var,ascending = True).reset_index(drop = True)  
                
        for j in target:                
            #rolling values
            df_temp[f'{j}_median_value'] = df_temp[j].rolling(window = windows_size, center = True).median()
            df_temp[f'{j}_diff_values'] = np.abs(df_temp[f'{j}_median_value'] - df_temp[j]) 
            df_temp[f'{j}_mads'] = df_temp[f'{j}_diff_values'].rolling(window = windows_size, center = True).median()

            df_temp[f'{j}_threshold'] = n_sigmas * df_temp[f'{j}_mads']

            #outliers based on threshold
            df_temp[f'{j}_outlier'] = df_temp[f'{j}_diff_values'] > df_temp[f'{j}_threshold']
            df_temp[f'imputed_{j}'] = np.where(df_temp[f'{j}_outlier'], df_temp[f'{j}_median_value'], df_temp[j])
            
            cols_to_drop.update([f'{j}_median_value',f'{j}_diff_values',f'{j}_mads',f'{j}_threshold'])
            
        result = pd.concat([result,df_temp])           
        
    result.reset_index(drop = True, inplace = True)
    
    df_temp = result.copy().drop(columns = list(cols_to_drop))
    
    return df_temp, result





