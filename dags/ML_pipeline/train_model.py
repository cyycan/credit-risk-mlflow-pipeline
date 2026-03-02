import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

"""This function is used to train a model
Arguments - 
    Input  : Training dataset model object
    Output : Model object
"""

def train_model(classifier, x_train, y_train, x_test, y_test):
    
    try: 
        classifier.fit(x_train, y_train)
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error occured in Training the model: {e}", exc_info = True)
        pass