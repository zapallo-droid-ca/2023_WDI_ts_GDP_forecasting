# ETL Load Functions

import sqlite3

##-- dim_country

def dim_country_load(wd, dbpath, table_name, data):  
    
    conn = sqlite3.connect(wd + dbpath)
    
    cursor = conn.cursor()  
    
    data = data.rename(columns = {'Country Name':'name','Country ISO3':'iso3','Country Code':'code','Long Name':'long_name',
                                                'Income Group':'income_group','Region':'region','Currency Unit':'currency_name'})
    
    cols_order = ['code','iso3','name','long_name','income_group','region','currency_name']
    
    data = data[cols_order].copy()
    
    # .csv files
    data.to_csv(wd + f'/data/final/{table_name}.csv.gz', index = False)
    
    # sql
    table_query = ''' CREATE TABLE  dim_country (
                      code VARCHAR(10) PRIMARY KEY,
                      iso3 VARCHAR(10),
                      name VARCHAR(30),
                      long_name VARCHAR(50),
                      income_group VARCHAR(30),
                      region VARCHAR(50),
                      currency_name VARCHAR(30)                    
                      )
                  '''                  
    try:
        cursor.execute(f"DROP TABLE {table_name}")  
        print(f'{table_name} has been eliminated from DB')
    except:
        print(f'{table_name} is not in the DB')
        
    finally:        
        cursor.execute(table_query)
        print(f'{table_name} has been created in the DB')       
        
    ## LOADING INTO SQL
    data = [tuple(x) for x in data.itertuples(index = False)]
    
    cursor.executemany(f'INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?)', data)
    print(f'{table_name} loaded')
    
    conn.commit()
    conn.close()               
          
    return print(f'{table_name} process finished')


def dim_calendar_load(wd, dbpath, table_name, data):  
    
    conn = sqlite3.connect(wd + dbpath)
    
    cursor = conn.cursor()    
    
    # .csv files
    data.to_csv(wd + f'/data/final/{table_name}.csv.gz', index = False)
    
    # sql
    table_query = ''' CREATE TABLE  dim_calendar (
                      time INTEGER PRIMARY KEY                  
                      )
                  '''                  
    try:
        cursor.execute(f"DROP TABLE {table_name}")  
        print(f'{table_name} has been eliminated from DB')
    except:
        print(f'{table_name} is not in the DB')
        
    finally:        
        cursor.execute(table_query)
        print(f'{table_name} has been created in the DB')       
        
    ## LOADING INTO SQL
    data = [tuple(x) for x in data.itertuples(index = False)]
    
    cursor.executemany(f'INSERT INTO {table_name} VALUES (?)', data)
    print(f'{table_name} loaded')
    
    conn.commit()
    conn.close()               
          
    return print(f'{table_name} process finished')


##-- ft_wdi
def ft_wdi_load(wd, dbpath, table_name, data):  
    
    conn = sqlite3.connect(wd + dbpath)
    
    cursor = conn.cursor()  
    
    data = data.rename(columns = {'economy':'country_iso3'})   
    
    # .csv files
    data.to_csv(wd + f'/data/final/{table_name}.csv.gz', index = False)
    
    # sql
    table_query = ''' CREATE TABLE  ft_wdi (
                      country_iso3 VARCHAR(10), 
                      time INTEGER, 
                      GDP_USD NUMERIC, 
                      GINI_per_capita NUMERIC,
                      'acapital_formation_%GDP' NUMERIC, 
                      'account_balance_%GDP' NUMERIC, 
                      'agriculture_%GDP' NUMERIC,
                      consumer_price_idx_b2010 NUMERIC, 
                      'consumption_%GDP' NUMERIC, 
                      'exports_%GDP' NUMERIC,
                      'foreign_investment_%GDP' NUMERIC, 
                      'gross_savings_%GDP' NUMERIC, 
                      'imports_%GDP' NUMERIC,
                      'industry_%GDP' NUMERIC, 
                      population NUMERIC, 
                      'remittances_%GDP' NUMERIC, 
                      'services_%GDP' NUMERIC,
                      GDP_USD_trend NUMERIC, 
                      GDP_USD_seasonal NUMERIC, 
                      GDP_USD_residual NUMERIC,
                      global_crisis BOOLEAN, 
                      GDP_USD_rank INTEGER,
                      population_rank INTEGER,
                      FOREIGN KEY (country_iso3) REFERENCES dim_country(iso3),
                      FOREIGN KEY (time) REFERENCES dim_calendar(time)
                      )
                  '''                  
    try:
        cursor.execute(f"DROP TABLE {table_name}")  
        print(f'{table_name} has been eliminated from DB')
    except:
        print(f'{table_name} is not in the DB')
        
    finally:        
        cursor.execute(table_query)
        print(f'{table_name} has been created in the DB')   
        
    ## LOADING INTO SQL
    data = [tuple(x) for x in data.itertuples(index = False)]
    
    cursor.executemany(f'INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data)
    print(f'{table_name} loaded')
    
    conn.commit()
    conn.close()               
          
    return print(f'{table_name} process finished')


##-- ft_tsd
def ft_tsd_load(wd, dbpath, table_name, data):  
    
    conn = sqlite3.connect(wd + dbpath)
    
    cursor = conn.cursor()  
    
    data = data.rename(columns = {'economy':'country_iso3'})   
    
    ## EXPORTING TO CSV
    # .csv files
    data.to_csv(wd + f'/data/final/{table_name}.csv.gz', index = False)
    
    ## TABLE CREATION
    # sql
    table_query = ''' CREATE TABLE  ft_tsd (
                      time INTEGER,
                      country_iso3 VARCHAR(10),                        
                      level NUMERIC, 
                      residual NUMERIC, 
                      seasonal NUMERIC, 
                      trend NUMERIC,                             
                      level_outlier NUMERIC, 
                      imputed_level NUMERIC, 
                      residual_outlier NUMERIC,                             
                      imputed_residual NUMERIC, 
                      global_crisis NUMERIC,
                      residual_lag_0 NUMERIC,                             
                      trend_lag_0 NUMERIC, 
                      level_lag_0 NUMERIC, 
                      seasonal_lag_0 NUMERIC, 
                      residual_lag_1 NUMERIC,
                      trend_lag_1 NUMERIC, 
                      level_lag_1 NUMERIC, 
                      seasonal_lag_1 NUMERIC, 
                      residual_lag_2 NUMERIC,                             
                      trend_lag_2 NUMERIC, 
                      level_lag_2 NUMERIC, 
                      seasonal_lag_2 NUMERIC,
                      FOREIGN KEY (country_iso3) REFERENCES dim_country(iso3),
                      FOREIGN KEY (time) REFERENCES dim_calendar(time)
                      )
                  '''                  
    try:
        cursor.execute(f"DROP TABLE {table_name}")  
        print(f'{table_name} has been eliminated from DB')
    except:
        print(f'{table_name} is not in the DB')
        
    finally:        
        cursor.execute(table_query)
        print(f'{table_name} has been created in the DB')      
    
    ## LOADING INTO SQL
    data = [tuple(x) for x in data.itertuples(index = False)]
    
    cursor.executemany(f'INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data)
    print(f'{table_name} loaded')
    
    conn.commit()
    conn.close()               
          
    return print(f'{table_name} process finished')


##-- other CSVs
def export_csv(wd, table_name, data):      
   
    data = data.rename(columns = {'economy':'country_iso3'})   
    
    ## EXPORTING TO CSV
    # .csv files
    data.to_csv(wd + f'/data/final/{table_name}.csv.gz', index = False)   
       
    return print(f'{table_name} process finished')