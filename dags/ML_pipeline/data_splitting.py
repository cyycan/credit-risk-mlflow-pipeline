import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split

import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

"""
This function is used to split dataset into training and validation dataset
Arguments -
    Input  : Dataset
    Output : x_test,  x_train, y_test, y_train
"""

def training_testing_dataset(data):
    
    try:

        df = data.drop(columns = ['SeriousDlqin2yrs'], axis=1)
        y  = data['SeriousDlqin2yrs']

        df_test, df_train, y_test, y_train = train_test_split(df, y, test_size = 0.8, random_state=42, stratify = y)
        
        train = pd.concat([df_train, y_train], axis=1)
        test  = pd.concat([df_test, y_test], axis=1)
        
        return df_test, df_train, y_test, y_train, train, test
    
    except Exception as e:
        logger.error(f"Error occured in splitting data into train and validation: {e}", exc_info = True)
        pass