





import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer

from bayes_opt import BayesianOptimization

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

import matplotlib.pyplot as plt

##-- General Functions

#Median Absolute Percentage Error (MAPE)
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


#Undo Series Diff
def rebuild_diffed(series, first_element_original):
    cumsum = series.cumsum()
    return cumsum.fillna(0) + first_element_original

def grx_train_test(X,Y,target_var):    
    # Visualize split C1
    fig,ax= plt.subplots(figsize=(12,3))
    kws = dict(marker='o')
    plt.plot(pd.concat([X[target_var],pd.Series(Y[target_var][:1])]), label='Train',**kws)
    plt.plot(Y[target_var], label='Test',**kws)
    plt.title('Train / Test Split')
    ax.legend(bbox_to_anchor=[1,1])
    plt.show()

#Recursive function
def predInter(forecast_result, vectFechTest, colsToBoost, test_c, lagsVolVenta, modelo):   
    
    for i in vectFechTest:    
        x = test_c[test_c.index == i] 
        if (lagsVolVenta == 1):        
            x.vol_venta_l1 = forecast_result.y.iloc[-1]

        elif (lagsVolVenta == 2):        
            x.vol_venta_l1 = forecast_result.y.iloc[-1]
            x.vol_venta_l2 = forecast_result.y.iloc[-2]
   
        elif (lagsVolVenta == 3):   
            x.vol_venta_l1 = forecast_result.y.iloc[-1]
            x.vol_venta_l2 = forecast_result.y.iloc[-2]
            x.vol_venta_l3 = forecast_result.y.iloc[-3]        

        forecast_ = modelo.predict(x[colsToBoost])

        forecast_result = forecast_result.append({'ds': i,
                                                  'y': forecast_[0],
                                                  'type': 'pred'}, ignore_index = True)

    forecast_result.set_index(forecast_result.ds, inplace = True)
    forecast_result.drop(columns = 'ds', inplace = True)
    forecast_result = forecast_result[forecast_result['type'] == 'pred']
    return forecast_result






##------ Parametros
n_iter = 50
n_estimators = 150
cv = 5
early_stopping_rounds = int(n_iter*0.2)
random_state = 2292

category_var = 'country_iso3'
target_var = 'level'
time_var = 'time'

number_of_diff = 1
train_size = 0.7


paramCorteNaive = 2
diferenciado = True

if (diferenciado == True): 
    strExportDiferenciado = 'diff'
else:
    strExportDiferenciado = 'sin_diff'

mape_loss = make_scorer(MAPE, greater_is_better=False)


df_base = pd.read_csv('data/final/ft_tsd.csv.gz')

df = df_base.copy()

categories_vector = df[category_var].unique()



##-- Step 1: Filtering
df = df[df[category_var] == 'USA'].sort_values(time_var, ascending = True).reset_index(drop = True).drop(columns = category_var).copy()
df.set_index(time_var, inplace = True)

##-- Step 2: Transformations
#- 2.1: Scaler
scaler = MinMaxScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)


#- 2.2: Diff
for i in range(1,number_of_diff+1):
    df = df.diff().iloc[i:,:]

#- 2.3: Diff
split_size = round(len(df)* train_size)
print(f'c1 train size: {split_size}')
print(f'c1 test size: {len(df.index) - split_size}')

# Split
X = df.iloc[:split_size]
Y = df.iloc[split_size:]



    






#Step X: Inverse Transformations
df = pd.DataFrame(scaler.inverse_transform(df), columns = df.columns)

















## Work Directory
wd = os.getcwd().replace('\src','\\')