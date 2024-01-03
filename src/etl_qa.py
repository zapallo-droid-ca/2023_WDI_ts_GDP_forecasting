

#Tiny QA: time range:       
def time_range_complete(df, category_var, time_var):            

    qa_a = df.shape[0] #number of observations
    
    qa_b = len(df[category_var].unique()) #categories
    
    qa_c = len(df[time_var].unique()) #timestamps   
    
    return (qa_a / qa_b) == qa_c