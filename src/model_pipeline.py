import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd

from src import model_ttp as mttp

##-- ARIMA
def arima_pipe(df, countries, prep_scaler, prep_diff, scaler_minmax, cols_exclude_preprocess, time_var, target_var, lags, 
               threshold, max_adf_iters, models_summary, models_results, category_var, train_size, predict_index, wd):
    
    ARIMA_done = []
    ARIMA_wrong = []
    
    model_name = 'ARIMA'
    
    for c_code, c_name in countries: 
        
        print(f'{c_name} {model_name} process started')
        
        df_temp = df[df[category_var] == c_code].reset_index(drop = True).sort_values(time_var, ascending = True)      
        
        try:
            data, aux_bj, scaler, first_element = mttp.ttp_preprocess(data = df_temp, prep_scaler = prep_scaler, prep_diff = prep_diff, scaler_minmax = scaler_minmax, 
                                                                      cols_exclude_preprocess = cols_exclude_preprocess, c_code = c_code, c_name = c_name, 
                                                                      time_var = time_var, target_var = target_var, lags = lags, threshold = threshold, 
                                                                      max_adf_iters = max_adf_iters)
                
            #Train/Test Split
            X, y, Xy = mttp.tt_split(data = data, train_size = train_size)
            
            data_model, model_summary, model = mttp.model_arima_ttp(X = X, y = y, Xy = Xy, aux_bj = aux_bj, first_element = first_element,
                                                                    predict_index = predict_index, scaler = scaler)
            
            
            models_summary, models_results = mttp.mod_sum(wd = wd + '/models/', data_model = data_model, model_summary = model_summary, 
                                                          model = model, c_code = c_code, prep_diff = prep_diff, prep_scaler = prep_scaler, 
                                                          scaler_minmax = scaler_minmax, model_name = model_name, models_summary = models_summary, 
                                                          models_results = models_results)        
            
            print(f'Forecast for {c_code}:  {c_name} using {model_name} model was done and data exported to models folder')
            
            ARIMA_done.append(c_code)
            
        except:
            print(f'Forecast for {c_code}:  {c_name} using {model_name} model was not done, model failed')
            
            ARIMA_wrong.append(c_code)      
    
                           
    with open(wd + '/logs/arima_wrong.txt', 'w') as f:
        for i in ARIMA_wrong:
            f.write('%s\n' % i)
    
    with open(wd + '/logs/arima_done.txt', 'w') as f:
        for i in ARIMA_done:
            f.write('%s\n' % i)
            
    return models_summary, models_results


##-- Holt-Winters
def howi_pipe(df, countries, cols_exclude_preprocess, time_var, target_var, lags, threshold, max_adf_iters, models_summary, iterHowi, typeHowi,
              models_results, category_var, train_size, predict_index, wd, scaler_minmax, prep_scaler, prep_diff):
    
    HOWI_done = []
    HOWI_wrong = []
    
    model_name = 'HOWI'
    
    for c_code, c_name in countries:      
        
        print(f'{c_name} {model_name} process started')
        
        df_temp = df[df[category_var] == c_code].reset_index(drop = True).sort_values(time_var, ascending = True)   
        
        try:  
            data, aux_bj, scaler, first_element = mttp.ttp_preprocess(data = df_temp, prep_scaler = prep_scaler, prep_diff = prep_diff, scaler_minmax = scaler_minmax, 
                                                                      cols_exclude_preprocess = cols_exclude_preprocess, c_code = c_code, c_name = c_name, 
                                                                      time_var = time_var, target_var = target_var, lags = lags, threshold = threshold, 
                                                                      max_adf_iters = max_adf_iters, ARIMA = False)
            
            #Train/Test Split
            X, y, Xy = mttp.tt_split(data = data, train_size = train_size)
            
            X.index = pd.date_range(start=str(X[time_var].min()), end=str(X[time_var].max()+1), freq='Y')
            y.index = pd.date_range(start=str(y[time_var].min()), end=str(y[time_var].max()+1), freq='Y')      
                
            
            data_model, model_summary, model  = mttp.model_howi_ttp(X = X, y = y, Xy = Xy, iterHowi = iterHowi, typeHowi = typeHowi, aux_bj = aux_bj, predict_index = predict_index, 
                                                                    first_element = first_element, scaler = scaler)
            
             
            models_summary, models_results = mttp.mod_sum(wd = wd + '/models/', data_model = data_model, model_summary = model_summary, 
                                                          model = model, c_code = c_code, prep_diff = prep_diff, prep_scaler = prep_scaler, 
                                                          scaler_minmax = scaler_minmax, model_name = model_name, models_summary = models_summary, 
                                                          models_results = models_results)         
           
            print(f'Forecast for {c_code}: {c_name} using {model_name} model was done and data exported to models folder')
            
            HOWI_done.append(c_code)
            
        except:            
            print(f'Forecast for {c_code}: {c_name} using {model_name} model was not done, model failed')
                    
            HOWI_wrong.append(c_code)      
    
                           
    with open(wd + '/logs/howi_wrong.txt', 'w') as f:
        for i in HOWI_wrong:
            f.write('%s\n' % i)
    
    with open(wd + '/logs/howi_done.txt', 'w') as f:
        for i in HOWI_done:
            f.write('%s\n' % i)
            
    return models_summary, models_results