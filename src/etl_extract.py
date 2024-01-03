# ETL Extract Functions

def wdi_extract(wd, range_min = 1991, range_max = 2022):    
    
    import yaml
    import pandas as pd
    import numpy as np
    import wbgapi as wb 
    import os    
      
    ## Config Data
    with open(wd + "/config/process/etl.yaml", "r") as f:
        config_file =  yaml.load(f, Loader=yaml.FullLoader)
        
    f.close()
    
    #Indicator codes
    wdi_data = pd.DataFrame(config_file['wdi'])
    
    #Paths
    wits_metadata_path = config_file['wits_metadata_path']
    
    ## Data Extraction
    #Metadata
    dim_country = pd.read_excel(wits_metadata_path)
    dim_country = dim_country[dim_country['Income Group'].isin(['Low income', 'Upper middle income', 'Lower middle income', 'High income'])].reset_index(drop = True)
    
    country_list = dim_country['Country ISO3'].unique()
    indicat_list = wdi_data['series'].unique()
    
    df_raw = wb.data.DataFrame(indicat_list, country_list, time=range(range_min, range_max), columns='time')
    df_raw.reset_index(drop = False, inplace = True)
    
    df = df_raw.copy()
    
    #Series Names
    df = df.merge(wdi_data[['series','col_name']], on = 'series', how = 'left')
    df = pd.melt(df, id_vars =  ['economy','series', 'col_name'] )
    df['variable'] = df['variable'].str.replace('YR','').astype(int)
    df['value_isna'] = df['value'].isna()
    df['value'] = df['value'].fillna(0)
    
    df.columns = df.columns.str.replace('variable','time')
    
    df.to_csv(wd + '/data/raw/data.csv.gz', index = False)
    
    return df, dim_country


def base_data(wd, nas_df = True):         
    #Reading again main data but now the extracted raw file
    import pandas as pd
    
    data = pd.read_csv(wd + '/data/raw/data.csv.gz')
    
    pivot_index = ['economy','time']
    pivot_columns = ['col_name']
    pivot_col_level = 1
    
    #Pivot and NAS dataset creation
    ft_nas = data.pivot(index = pivot_index, columns = pivot_columns, values = ['value_isna']).reset_index(col_level = pivot_col_level)
    ft_nas.columns = ft_nas.columns.droplevel()
    
    ft_wdi = data.pivot(index = pivot_index, columns = pivot_columns, values = ['value']).reset_index(col_level = pivot_col_level)
    ft_wdi.columns = ft_wdi.columns.droplevel()
    
    if nas_df:
        return ft_wdi, ft_nas
    else:
        return ft_wdi






