# ETL Transform Functions

def base_data(wd, nas_df = True):          
    
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