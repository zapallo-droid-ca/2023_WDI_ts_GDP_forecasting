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
    
    df.to_csv('./data/raw/data.csv.gz')
    
    return df, dim_country







