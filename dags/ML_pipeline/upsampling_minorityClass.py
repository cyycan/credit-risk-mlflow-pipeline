from imblearn.over_sampling import SMOTE
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

"""
This function is used to upsample the minority class using SMOTE technique from synthetic sampling
Arguments - 
    Input  : Featured Engineered df
    Output : Upsampled df 
"""

def upsampling_class(train, test, val, upsampling=None):
    
    try:
        if upsampling : 
            x_train = train.drop(columns = ['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            y_train = train['SeriousDlqin2yrs']

            test_x = test.drop(columns = ['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            test_y = test['SeriousDlqin2yrs']

            val_x = val.drop(columns = ['CustomerID'], axis=1)

            smote = SMOTE(sampling_strategy = 'minority',k_neighbors = 2,random_state=42)
            os_data_X,os_data_y = smote.fit_resample(x_train,y_train)

            return os_data_X, os_data_y, test_x, test_y, val_x
        
        else:
            x_train = train.drop(columns = ['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            y_train = train['SeriousDlqin2yrs']

            test_x = test.drop(columns = ['CustomerID', 'SeriousDlqin2yrs'], axis=1)
            test_y = test['SeriousDlqin2yrs']

            val_x = val.drop(columns = ['CustomerID'], axis=1)
            
            return x_train, y_train, test_x, test_y, val_x
            

    except Exception as e:
        logger.error(f"Error occured in Upsampling of minority class: {e}", exc_info = True)
        pass