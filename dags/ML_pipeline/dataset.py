import pandas as pd
import numpy as np 

import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

"""
This function is used to read data files from a certain path
Arguments - 
    Output : train, test
"""


train_path      = "../input/cs-training.csv"
validation_path = "../input/cs-test.csv"

def read_data():
    
    try:
        train = pd.read_csv(train_path)

        if 'Unnamed: 0' in train.columns.tolist():
            train = train.rename(columns={'Unnamed: 0' : 'CustomerID'})

        val  = pd.read_csv(validation_path)

        if 'Unnamed: 0' in val.columns.tolist():
            val = val.rename(columns={'Unnamed: 0' : 'CustomerID'})
        val.drop(columns=['SeriousDlqin2yrs'], axis=1, inplace=True)
        
        return train,val
        
    except Exception as e:
        logger.error(f"Error occured in reading data files: {e}", exc_info = True)
        pass  
