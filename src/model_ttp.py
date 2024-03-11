import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer

from bayes_opt import BayesianOptimization

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb


from src import model_preprocess as mprep

def tt_split(data, train_size):    
        ###-- Train / Test Split
        split_size = round(len(data)* train_size)
        
        # Split
        X = data.iloc[:split_size]
        y = data.iloc[split_size:]
        Xy = data
        
        #Labeling main dataset
        data['tts_flag'] = data.index
        data['tts_flag'] = data['tts_flag'].apply(lambda x: 'Train' if x in X.index else 'Test') 
        
        return X, y, Xy
    
    
    
def ttp_preprocess(data, prep_scaler, prep_diff, scaler_minmax, cols_exclude_preprocess, c_code, c_name, time_var, target_var, 
                   lags, threshold, max_adf_iters, ARIMA = True):
    
    if prep_scaler:
        data, scaler = mprep.scaled_values(df = data, scaler_minmax = True, cols_exclude_preprocess = cols_exclude_preprocess)
    else:
        scaler = None
    
    aux_bj = mprep.bj_autoregressive_values(df = data, c_code = c_code, c_name = c_name, time_var = time_var, target_var = target_var, lags = lags, 
                                            threshold = threshold, max_adf_iters = max_adf_iters) 
    
    if ARIMA:  
        if aux_bj['i_order'].values[0] == 0: #To keep all the series with the same structure d==0 fail for ARIMA
            aux_bj['i_order'] = [1]

    if prep_diff:
        data,first_element = mprep.target_diff(df = data, autoregressive_vals = aux_bj, target_var = target_var)
    else:
        data['target'] = data[target_var]
        first_element = None        

    return data, aux_bj, scaler, first_element



def mod_sum(wd , data_model, model_summary, model, c_code, prep_diff, prep_scaler, scaler_minmax, model_name, 
            models_summary, models_results):
    
        data_model['country_iso3'] = c_code
        model_summary['country_iso3'] = c_code
        model_summary['diff'] = prep_diff
        if prep_scaler and scaler_minmax:
            model_summary['scaler'] = 'minmax'
        elif prep_scaler:
            model_summary['scaler'] = 'std'
        else:
            model_summary['scaler'] = 'none'
        
        model_summary.to_csv(wd + f'/summary/raw/{c_code}_{model_name}.csv.gz', index = False)
        data_model.to_csv(wd + f'/data/raw/{c_code}_{model_name}.csv.gz', index = False)
        
        models_summary.append(model_summary.to_dict('records'))
        models_results.append(data_model.to_dict('records'))   
        
        return models_summary, models_results
    
    
##-- ARIMA    
def model_arima_ttp(X, y, Xy, aux_bj, predict_index, prep_diff = True, first_element = None, prep_scaler = True , scaler = None):
    
    '''
    prep_diff: True (default value) - Meaning that the given data in X, y and Xy was diff during the preprocessing
    prep_scaler: True (default value) - Meaning that the given data in X, y and Xy was scaled during the preprocessing
    scaler: scaler object from preprocessing if is necessary
    aux_bj: object from preprocessing with Box and Jenkins attributes
    first_element: first element of series getting from preprocessing - differenciation
    
    '''
    
    model_summary = []
    model_results = {}
    
    ###----TRAINING
    ### AUTOREGRESSIVE MODELS
    vectAR = list(range(1,max(aux_bj.ar_order[0])+1))
    vectMA = list(range(1,max(aux_bj.ma_order[0])+1))
    #vectAR = aux_bj.ar_order[0]    
    #vectMA = aux_bj.ma_order[0]
    
    if 0 in vectAR:
        vectAR.remove(0)
    if 0 in vectMA:
        vectMA.remove(0) 
    
    d = aux_bj.i_order[0]
    
    ##Recursive Model    
    for p in vectAR:
        for q in vectMA:        
            try:
                #Train
                model = ARIMA(X['target'], order = (p,d,q))
                model = model.fit()        
                
                X_rec = X['target'].values #To iter over and append each iter result
                y_pred = [] #To be filled with predictions
                
                for i in range(0,len(y)):
                    pred = model.apply(X_rec[i:]).forecast()
                    X_rec = np.append(X_rec, pred)
                    y_pred.append(pred[0])
                    
                y_pred = pd.Series(index = y.index, data = y_pred)
                
                #Metrics
                mape = mean_absolute_percentage_error(y_pred, y['target'])
                
                model_summary.append({'model': 'ARIMA',
                                       'exp': f'{p},{d},{q}',
                                       'mape': mape,
                                       'object': model})
                
                model_results[f'{p},{d},{q}'] = y_pred.values
            except:
                print(f'The model failed in Iter p:{p} d:{d} q:{q}')
    
    model_summary = pd.DataFrame(model_summary)
    model_summary = model_summary[model_summary['model'] == 'ARIMA'].reset_index(drop = True)
    
    exp_min_mape = model_summary[model_summary['mape'] == min(model_summary.mape)]['exp'].values[0]
    model_results = model_results[exp_min_mape]
    model = model_summary[model_summary['exp'] == exp_min_mape]['object'].reset_index(drop = True)[0]
    
    data_model = Xy.copy()
    data_model['y_pred'] = data_model['target']
    data_model.iloc[-len(y):,-1] = model_results    
    data_model['exp'] = f'{p},{d},{q}'
    
    ###----PRODUCTION
    del(X_rec,y_pred,mape)
    
    X_pred = Xy['target'].values
    y_pred = []
    
    for i in range(0,len(predict_index)):
        pred = model.apply(X_pred[i:]).forecast()
        X_pred = np.append(X_pred, pred)
        y_pred.append(pred[0])
        
    data_model = pd.concat([data_model,pd.DataFrame({'time':predict_index,
                                                      'level': [np.nan] * len(predict_index),
                                                      'target': [np.nan] * len(predict_index),
                                                      'tts_flag': ['Predict']  * len(predict_index),
                                                      'y_pred': y_pred})])
      
    #Des-diferenciamos la variable objetivo
    if prep_diff and first_element != None:    
        data_model[['level','target','y_pred']] = mprep.rebuild_diffed(data_model[['level','target','y_pred']],first_element)#.values
    
    #Desescalamos
    if prep_scaler and scaler != None:        
        cols = ['level','target','y_pred']   
        #cols = ['y_pred']   
        data_model.loc[data_model['tts_flag'] != 'Predict',cols] = pd.DataFrame(scaler.inverse_transform(data_model.loc[data_model['tts_flag'] != 'Predict',cols]), columns=cols)
        data_model.loc[data_model['tts_flag'] == 'Predict',cols] = pd.DataFrame(scaler.inverse_transform(data_model.loc[data_model['tts_flag'] == 'Predict',cols]), columns=cols)
                      
        cols = ['level','target']
        data_model.loc[data_model['tts_flag'] == 'Predict',cols] = np.nan

    return data_model, model_summary, model


##-- Holt-Winters
def model_howi_ttp(X, y, Xy, iterHowi, typeHowi, aux_bj, predict_index, prep_diff = True, first_element = None, prep_scaler = True , scaler = None):
    
    model_name = 'HOWI'
    
    model_summary = []
    model_results = {}
    
    seasonal_periods = range(2,int(np.floor(len(X.index)/2))) #Iterating with top in half length 
    
    iters = len(iterHowi)**3 * len(typeHowi)**2 * len(seasonal_periods)
    
    counting = 0

    for i in iterHowi:
        for j in iterHowi:
            for k in typeHowi:
                for l in typeHowi:
                    for m in iterHowi:
                        for n in seasonal_periods:
                            #Train
                            
                            model = ExponentialSmoothing(X['target'], trend = k, seasonal = l, seasonal_periods = n)
                            model = model.fit(smoothing_level= i, smoothing_trend = j, smoothing_seasonal = m, optimized=False)
                            
                            X_rec = X['target'].values #To iter over and append each iter result
                            y_pred = [] #To be filled with predictions
                            
                            for o in range(len(y)):
                                pred = model.forecast(1) #Using the model to predict one step        
                                X_rec = np.append(X_rec, pred)                                
                                y_pred.append(pred[0])
                                
                                #Updating the model
                                model = ExponentialSmoothing(X_rec[o:], trend = k, seasonal = l, seasonal_periods = n)
                                model = model.fit(smoothing_level= i, smoothing_trend = j, smoothing_seasonal = m, optimized=False)
                                
                            y_pred = pd.Series(index = y.index, data = y_pred)
                                
                            #Metrics
                            mape = mean_absolute_percentage_error(y_pred, y['target'])
                            
                            exp_tag = f's_level:{i}, s_trend:{j}, s_seasonal:{m}, trend:{k}, level:{l}, s_periods:{n}'
                            
                            model_summary.append({'model': model_name,
                                                   'exp': exp_tag,
                                                   'mape': mape,
                                                   'object': model})
                            
                            model_results[exp_tag] = y_pred.values         
                            
                            counting +=1
                                                      
                            print(f'iter {counting} from {iters} ended {np.round((counting/iters) * 100,1)}% Completed')
                            
    model_summary = pd.DataFrame(model_summary)
    model_summary = model_summary[model_summary['model'] == model_name].reset_index(drop = True)
    
    exp_min_mape = model_summary[model_summary['mape'] == min(model_summary.mape)]['exp'].values[0]
    model_results = model_results[exp_min_mape]
    model = model_summary[model_summary['exp'] == exp_min_mape]['object'].reset_index(drop = True)[0]
    
    data_model = Xy.copy()
    data_model['y_pred'] = data_model['target']
    data_model.iloc[-len(y):,-1] = model_results  
    data_model['exp'] = f's_level:{i}, s_trend:{j}, s_seasonal:{m}, trend:{k}, level:{l}, s_periods:{n}'
    
    ###----PRODUCTION
    del(X_rec,y_pred,mape)
    
    X_pred = Xy['target'].values
    y_pred = []
    
    for i in range(len(predict_index)):
        pred = model.forecast(1) #Using the model to predict one step      
        X_pred = np.append(X_pred, pred)
        y_pred.append(pred[0])
        
        #Updating the model
        model = ExponentialSmoothing(X_pred[i:], trend = k, seasonal = l, seasonal_periods = n)
        model = model.fit(smoothing_level= i, smoothing_trend = j, smoothing_seasonal = m, optimized=False)
        
    data_model = pd.concat([data_model,pd.DataFrame({'time':predict_index,
                                                      'level': [np.nan] * len(predict_index),
                                                      'target': [np.nan] * len(predict_index),
                                                      'tts_flag': ['Predict']  * len(predict_index),
                                                      'y_pred': y_pred})])
    
    #Des-diferenciamos la variable objetivo
    if prep_diff and first_element != None:    
        data_model[['level','target','y_pred']] = mprep.rebuild_diffed(data_model[['level','target','y_pred']],first_element)#.values
    
    #Desescalamos
    if prep_scaler and scaler != None:        
        cols = ['level','target','y_pred']   
        data_model.loc[data_model['tts_flag'] != 'Predict',cols] = pd.DataFrame(scaler.inverse_transform(data_model.loc[data_model['tts_flag'] != 'Predict',cols]), columns=cols)
        data_model.loc[data_model['tts_flag'] == 'Predict',cols] = pd.DataFrame(scaler.inverse_transform(data_model.loc[data_model['tts_flag'] == 'Predict',cols]), columns=cols)
        
        cols = ['level','target']
        data_model.loc[data_model['tts_flag'] == 'Predict',cols] = np.nan

    return data_model, model_summary, model        



    #Des-diferenciamos la variable objetivo
    if prep_diff and first_element != None:    
        data_model[['level','target','y_pred']] = mprep.rebuild_diffed(data_model[['level','target','y_pred']],first_element)#.values
    
    #Desescalamos
    if prep_scaler and scaler != None:        
        cols = ['level','target','y_pred']   
        data_model.loc[data_model['tts_flag'] != 'Predict',cols] = pd.DataFrame(scaler.inverse_transform(data_model.loc[data_model['tts_flag'] != 'Predict',cols]), columns=cols)
        data_model.loc[data_model['tts_flag'] == 'Predict',cols] = pd.DataFrame(scaler.inverse_transform(data_model.loc[data_model['tts_flag'] == 'Predict',cols]), columns=cols)
        
        cols = ['level','target']
        data_model.loc[data_model['tts_flag'] == 'Predict',cols] = np.nan

    return data_model, model_summary, model