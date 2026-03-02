import pandas as pd 
from scipy import stats, special
from scipy.stats import skew
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def SkewMeasure(df):
    nonObjectColList = df.dtypes[df.dtypes != 'object'].index
    skewM = df[nonObjectColList].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
    skewM=pd.DataFrame({'skew':skewM})
    return skewM[abs(skewM)>0.5].dropna()

"""
This feature is used to do Scaling of the features
Arguments - 
    Input  : Upsampled dataframes
    Output : Scaled features dataframes
"""

def scaling_features(train, test, val, scaling=None):
    
    try:    
        if scaling:
            skewM = SkewMeasure(train)
            for i in skewM.index:
                train[i] = special.boxcox1p(train[i],0.15) #lambda = 0.15
                test[i]  = special.boxcox1p(test[i],0.15) #lambda = 0.15
                val[i]  = special.boxcox1p(val[i],0.15) #lambda = 0.15

            return train, test, val
        
        else:
            return train, test, val
  
    except Exception as e:
        logger.error(f"Error occured in Features Scaling: {e}", exc_info = True)
        pass