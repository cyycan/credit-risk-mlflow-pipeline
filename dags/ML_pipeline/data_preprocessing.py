import pandas as pd 
import numpy as np 

import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

"""
This function will clean data entries where rows need to be drop, impute missing values
Arguments - 
    Input  : Train, Test, Validation Datasets and the required columns
    Output :Cleaned Train, Test, Validation Datasets 
"""

def data_preprocessing(train, test, val, col1, col2, col3, col4, col5, col6, col7, col8, target):
    
    try:
        # Cleaning data entries and treating outliers
        
        train       = train[-((train[col5] > train[col5].quantile(0.95)) & (train[target] == train[col4]))]
        test        = test[-((test[col5] > test[col5].quantile(0.95)) & (test[target] == test[col4]))]

        train       = train[train[col6]<=10]
        test        = test[test[col6]<=10]
        val         = val[val[col6]<=10]
        
        train.loc[train[col7] == 0, col7] = train.age.mode()[0]
        test.loc[test[col7] == 0, col7]   = test.age.mode()[0]
        val.loc[val[col7] == 0, col7]     = val.age.mode()[0]
    
        # Imputing Missing values
        train[col4].fillna(train[col4].median(), inplace=True)
        test[col4].fillna(test[col4].median(), inplace=True)
        val[col4].fillna(val[col4].median(), inplace=True)
        
        train[col8].fillna(0, inplace = True)
        test[col8].fillna(0, inplace = True)
        val[col8].fillna(0, inplace = True)
        
        return train, test, val
    
    except Exception as e:
        logger.error(f"Error occured in Data Preprocessing: {e}", exc_info = True)
        pass   