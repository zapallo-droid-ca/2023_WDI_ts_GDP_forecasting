### Project ETL

## Libraries
import pandas as pd
import numpy as np
import os
import requests
import json
import sqlite3

## Work Directory
wd = os.getcwd()

from src import etl_extract

data, dim_country = etl_extract.wdi_extract(wd, range_min = 1991, range_max = 2022)


#Pivot and NAS dataset creation
ft_nas = data.pivot(index = ['economy','time'], columns = ['col_name'], values = ['value_isna']).reset_index(col_level = 1)
ft_nas.columns = ft_nas.columns.droplevel()

ft_wdi = data.pivot(index = ['economy','time'], columns = ['col_name'], values = ['value']).reset_index(col_level = 1)
ft_wdi.columns = ft_wdi.columns.droplevel()

#Dummies
ft_wdi['finantial_crisis'] = (ft_wdi.time >= 2007) & (ft_wdi.time <= 2008)
ft_wdi['pandemic'] = (ft_wdi.time >= 2020) & (ft_wdi.time <= 2023)

#TSD







#INCLUIR: TSD


