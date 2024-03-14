
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scaled_values(df, scaler_minmax, cols_exclude_preprocess):
    
    data = df.copy()
    
    #cols = data.drop(columns = cols_exclude_preprocess).columns    
    
    if scaler_minmax:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    #data[cols] = scaler.fit_transform(data[cols])
    data['target'] = scaler.fit_transform(data[['target']])
    
    return data, scaler

        

def bj_autoregressive_values(df, c_code, c_name, time_var, target_var, lags, threshold, max_adf_iters):
    
    results = pd.DataFrame()

    # Only for Classic models
    df_temp = df[[time_var,target_var]]
    data = df_temp.set_index(time_var)
    
    #Stationary Moddeling          
    test = adfuller(data, regresults = True)
    test = pd.DataFrame([{'category_code':c_code,'category_name':c_name,'adf':test[0],'pvalue':test[1],'diff':0,'stationary': test[1] <= 0.05}])
    
    iters = 0
    
    while test.stationary.values[0] == False:
        iters += 1
        data = data.diff().dropna()
        test = adfuller(data, regresults = True)
        test = pd.DataFrame([{'category_code':c_code,'category_name':c_name,'adf':test[0],'pvalue':test[1],'diff':iters,'stationary': test[1] <= 0.05}])
    
        if iter == max_adf_iters: break
    
    #Functions
    acf_values = acf(data, nlags=lags)
    pacf_values = pacf(data, nlags=lags)
    
    
    # Extract AR components
    ar_order = [i for i, val in enumerate(pacf_values) if abs(val) > threshold]
    
    # Extract MA components
    ma_order = [i for i, val in enumerate(acf_values) if abs(val) > threshold]
    
    # Extractin I component
    i_order = iters
    
    series_order = [{'category_code':c_code,'category_name':c_name,'ar_order':ar_order,'i_order':i_order,'ma_order':ma_order}]
    
    results = pd.concat([results, pd.DataFrame(series_order)])
        
    return results


def target_diff(df, autoregressive_vals, target_var):
    
    first_element = df.iloc[0,][target_var]
    
    if autoregressive_vals.i_order.values[0] != 0:    
        for i in range(0, autoregressive_vals.i_order.values[0]):
            df['target'] = df[target_var].diff()
    else:
        df['target'] = df[target_var]
    
    df = df.dropna().reset_index(drop = True)
    return df,first_element

def rebuild_diffed(series, autoregressive_vals, first_element_original):
    
    if autoregressive_vals.i_order.values[0] != 0:    
        for i in range(0, autoregressive_vals.i_order.values[0]):
            cumsum = series.cumsum().fillna(0) + first_element_original
    else:
        cumsum = series.cumsum().fillna(0) + first_element_original        
     
    return cumsum








