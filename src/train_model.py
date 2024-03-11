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