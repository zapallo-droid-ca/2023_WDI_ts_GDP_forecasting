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
def tsDecomposition(data, index_frequency, period, target, category, sub_category = None, sdModel = 'additive', scale_data = False, two_sided = False):
    """
    If scale_data == True, the function scale the target variable by category using Z-Score normalization.

    The function should receive a pandas dataframe aggregated with the category and sub_category vars and with the timestamp in the index.

    Where:
        * index_frequency (str): parameter for .asfreq() -- 'YS: Annual Std', 'm'
        * period (int): parameter period of seasonal_decompose (YS: 12, Annual: 12)
        * sdModel (str): parameter for statsmodels seasonal_decompose -- 'additive' / 'multiplicative'
        * two_sided (bool): parameter for statsmodels seasonal_decompose

    This functions return 2 pandas dataframes with:
        1. all components scalated, the level scalated and the level with original values
        2. measures used for scaling
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
    df_level_scaled = pd.DataFrame()

    scaler_measures = []

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

        if scale_data:
            #-Scaling
            series_statistics = {'category':cat,'sub_category':subcat,'mean':df_temp[target].mean(),'std':df_temp[target].std()} 

            series_level = df_temp[target].copy()

            scaler = StandardScaler()
            df_temp[target] = scaler.fit_transform(df_temp[target].values.reshape(-1,1))
            series_level_scaled = df_temp[target].copy()
        else:
            series_level = df_temp[target].copy()

        #-Time Series Decomposition
        sdDF = seasonal_decompose(df_temp[target], model = sdModel, period = period, two_sided = two_sided)

        if scale_data:     
            temp_level_scaled = {'date': series_level_scaled.index,
                                 'value': series_level_scaled.values,
                                 category:cat,
                                 sub_category:subcat,
                                 'target':target,
                                 'component':'level_scaled'}
            
        temp_level = {'date': series_level.index,
                      'value': series_level.values,
                      category:cat,
                      sub_category:subcat,
                      'target':target,
                      'component':'level_original'}

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
        if scale_data:
            df_level_scaled = pd.concat([df_level_scaled,pd.DataFrame(temp_level_scaled)])
            scaler_measures.append(series_statistics)  

        if scale_data: 
            del(temp_trend,temp_seasonal,temp_resid,temp_level,temp_level_scaled, sdDF, scaler, series_statistics)
        else:
            del(temp_trend,temp_seasonal,temp_resid,temp_level, sdDF)  
        print(f'{target} by {cat} done for subcategory {subcat}', end = '\r') 

    if scale_data:
        scaler_measures = pd.DataFrame(scaler_measures)
        df_timeSeriesDecomp = pd.concat([df_trend,df_seasonal,df_residual,df_level,df_level_scaled]).reset_index(drop = True)
        components_pivot = ['level_original','level_scaled','trend','seasonal','residual']
    else:
        df_timeSeriesDecomp = pd.concat([df_trend,df_seasonal,df_residual,df_level]).reset_index(drop = True)
        components_pivot = ['level_original','trend','seasonal','residual']   

    if sub_category == None:
        df_timeSeriesDecomp = df_timeSeriesDecomp[df_timeSeriesDecomp['component'].isin(components_pivot)].pivot_table(index = ['date',category], columns = 'component', values = 'value', aggfunc = 'sum').reset_index()
    else:
        df_timeSeriesDecomp = df_timeSeriesDecomp[df_timeSeriesDecomp['component'].isin(components_pivot)].pivot_table(index = ['date',category,sub_category], columns = 'component', values = 'value', aggfunc = 'sum').reset_index()
    df_timeSeriesDecomp.columns = df_timeSeriesDecomp.columns.rename('')
    df_timeSeriesDecomp.sort_values('date', ascending = True, inplace = True)
    
    if scale_data:
        print(f'process ended, df_timeSeriesDecomp: {df_timeSeriesDecomp.shape} and scaler_measures: {scaler_measures.shape}')        
    else:
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

    if scale_data:
        return df_timeSeriesDecomp, scaler_measures
    else:
        return df_timeSeriesDecomp

##-- DYNAMIC TIME WARPING MATRIX
def dtw_matrix_funct(data, index, columns, values, category = None):
    """
    Aplication of DTW in two time series (x1 and x2)
    
    Returns two dictionary's with the matrix or arrays (in case of multiple values in the category) with the category and each array or matrix and an equivalent object with the index evaluated

    Please, introduce as data argument the dataset in time series format, the function will pivot it to transform it into a classification problem.
    """
    if category == None:
        categories = pd.Series([index]).unique()
    else:
        categories = data[category].unique()

    matrix_dicts = {}
    index_dicts = {}

    if len(categories) == 1:
        categories = categories.item() #extracting the tex from the array

        print(f'working with the category {categories} of {values} as unique category in the dataframe', end = '\r')
        if category == None:
            X = data[[index,columns,values]].reset_index(drop = True).sort_values(columns, ascending = True)
        else:            
            X = data[data[category] == categories][[index,columns,values]].reset_index(drop = True).sort_values(columns, ascending = True)
        X = X.pivot(index = index, columns = columns, values = values)

        series = X.shape[0]
        dtwMatrix = np.zeros((series,series))
        indexMatrix = []

        for ts1 in range(series):
            for ts2 in range(ts1, series):
                dtwValue, path = fastdtw(X.iloc[ts1].values, X.iloc[ts2].values)
                dtwMatrix[ts1,ts2] = dtwValue 
                dtwMatrix[ts2,ts1] = dtwValue

            indexMatrix.append(X.iloc[ts1].name)

        matrix_dicts[categories] = dtwMatrix
        index_dicts[categories] = indexMatrix

    else:
        for cat in categories:
            print(f'working with the category {cat} of {values}', end = '\r')

            X = data[data[category] == cat][[index,columns,values]].reset_index(drop = True).sort_values(columns, ascending = True)
            X = X.pivot(index = index, columns = columns, values = values)

            series = X.shape[0]
            dtwMatrix = np.zeros((series,series))
            indexMatrix = []

            for ts1 in range(series):
                for ts2 in range(ts1, series):
                    dtwValue, path = fastdtw(X.iloc[ts1].values, X.iloc[ts2].values)
                    dtwMatrix[ts1,ts2] = dtwValue 
                    dtwMatrix[ts2,ts1] = dtwValue

                indexMatrix.append(X.iloc[ts1].name)

            matrix_dicts[cat] = dtwMatrix
            index_dicts[cat] = indexMatrix

    print(f'process finished, {len(matrix_dicts)} DTW matrix created and {len(index_dicts)} index lists created', end = '\r')

    return matrix_dicts, index_dicts


##-- OPTIMAL CLUSTER NUMBERS: KMEANS
def clustering_kmeans_multi(x, y, categories, early_stop_yield, randomStateValue):
    """
        This functions takes as imput the objects returned by dtw_matrix_funct, two dict of arrays,
        (not a dataset), and return two new objects with the clusters id for each index in the number of experimented
        clusters.

        x (dict): dict with arrays where each keys are categories to be iterated, first object returned by dtw_matrix_funct
        y (dict): dict with arrays where each keys are categories to be iterated, second object returned by dtw_matrix_funct
        categories (list): categories to be iterated
        early_stop_yield (float): 0.1 - 0.9 to calculate the early stop.
        randomStateValue (int): parameter of KMeans
    """
    def mode(x):
        values, counts = np.unique(x, return_counts=True)
        m = counts.argmax()
        return values[m]

    result = {}
    clusters_tests = []   

    for cat in categories:
        X = x[cat]
        Y = y[cat]
        maxClusterNumbers = X.shape[0]     

        best_score_s = 0
        best_score_c = 0
        best_score_d = float('inf') #min value of function is 0

        #To avoid overfitting lets work in a stop function
        iter_stop = np.ceil(maxClusterNumbers * early_stop_yield)

        ss_counter = 0
        sc_counter = 0
        sd_counter = 0

        #Saving labels
        clusters_labels = {}
        clusters_list = [0,0,0]

        result_temp = pd.DataFrame()

        print(f'working with {cat} and {maxClusterNumbers} observations and {iter_stop} stable iters break', end = '\r')

        for clusters in range(2,maxClusterNumbers):

            kmeans = KMeans(n_clusters = clusters, random_state = randomStateValue).fit(X)
            labels = kmeans.labels_      

            clusters_labels[clusters] = labels  

            score_s = silhouette_score(X, labels)
            score_c = calinski_harabasz_score(X, labels)
            score_d = davies_bouldin_score(X, labels)

            #scores_s_list.append(score_s)
            #scores_c_list.append(score_c)
            #scores_d_list.append(score_d)

            if score_s > best_score_s:
                best_score_s = score_s
                clusters_s = clusters
                ss_counter = 0        
                clusters_list[0] = clusters
            else:        
                ss_counter += 1 #Adding 1 to counter for early stop
            
            if score_c > best_score_c:
                best_score_c = score_c 
                clusters_c = clusters
                sc_counter = 0        
                clusters_list[1] = clusters
            else:        
                sc_counter += 1 #Adding 1 to counter for early stop

            if score_d < best_score_d:
                best_score_d = score_d 
                clusters_d = clusters  
                sd_counter = 0         
                clusters_list[2] = clusters
            else:        
                sd_counter += 1 #Adding 1 to counter for early stop

            if ss_counter == iter_stop or sc_counter == iter_stop or sd_counter == iter_stop:


                print(f'{iter_stop} iters without change reached ({early_stop_yield} yield), iters without change by measure -> sil: {ss_counter}, cal: {sc_counter}, dav: {sd_counter}', end = '\n')
                break     
        
        suggested_clusters = mode([clusters_s,clusters_c,clusters_d])

        #Experiments
        clusters_tests.append({'category': cat,
                               'clusters_silhouette': clusters_s,
                               'clusters_calinski_Harabasz': clusters_c,
                               'clusters_davies_Bouldin': clusters_d,
                               'suggested_clusters': suggested_clusters,
                               #'score_silhouette': best_score_s,
                               #'score_calinski_Harabasz': best_score_c,
                               #'score_davies_Bouldin': best_score_d
                            })   
        
        #Results - Labels
        clusters_list = list(set(clusters_list))      

        result_temp['index'] = Y

        for clusters in clusters_list:
            result_temp[f'labels_{clusters}_clusters'] = clusters_labels[clusters]

        result_temp[f'suggested_clusters'] = clusters_labels[suggested_clusters]

        result[cat] = result_temp

    clusters_tests = pd.DataFrame(clusters_tests)    

    print('process finished','\n')

    return result, clusters_tests


##-- RESHAPING DATA FOR KMEANS (WHEN NOT USING DTW)
def ts_component_kmeans_preproc(data, index, columns, values, category):
    """
    Returns two dictionary's with the matrix or arrays (in case of multiple values in the category) with the category and each array or matrix and an equivalent object with the index evaluated

    Please, introduce as data argument the dataset in time series format, the function will pivot it to transform it into a classification problem.

    The function was build to use as imput of clustering_kmeans_multi when not using DTW
    """
    categories = data[category].unique()

    matrix_dicts = {}
    index_dicts = {}

    if len(categories) == 1:
        categories = categories.item() #extracting the tex from the array

        print(f'working with the category {categories} of {values} as unique category in the dataframe', end = '\r')

        X = data[data[category] == categories][[index,columns,values]].reset_index(drop = True).sort_values(columns, ascending = True)
        X = X.pivot(index = index, columns = columns, values = values)

        matrix_dicts[categories] = X.values
        index_dicts[categories] = list(X.index.values)

    else:
        for cat in categories:
            print(f'working with the category {cat} of {values}', end = '\r')

            X = data[data[category] == cat][[index,columns,values]].reset_index(drop = True).sort_values(columns, ascending = True)
            X = X.pivot(index = index, columns = columns, values = values)

            matrix_dicts[cat] = X.values
            index_dicts[cat] = list(X.index.values)

    print(f'process finished, {len(matrix_dicts)} matrix created and {len(index_dicts)} index lists created', end = '\r')

    return matrix_dicts, index_dicts



##-- HAMPEL FILTER FOR TS OUTLIERS
def hampel_filter(data, target, index, windows_size, n_sigmas = 3):
    """
    Apply the Hampel filter to a time series.
    
    Parameters:
    input_series (pd.Series): The input time series to filter.
    window_size (int): The size of the rolling window (odd integer).
    n_sigmas (int): The number of standard deviations to use as the threshold.
    
    Returns:
    pd.DataFrame: The filtered data
    """
    result = data.copy()

    #rolling values
    median_value = data[target].rolling(window = windows_size, center = True).median()
    std_value = data[target].rolling(window = windows_size, center = True).std()

    threshold = n_sigmas * std_value

    #outliers based on threshold
    result['outlier'] = np.abs(data['residual'] - 1) > threshold
    result[f'imputed_{target}_values'] = median_value[result['outlier']]
    result[f'imputed_{target}_values'] = np.where(result['outlier'], result[f'imputed_{target}_values'], result[target])

    return result





