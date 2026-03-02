import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

"""This function is used to predict values using the trained model
Arguments - 
    Input  : Model trained on training dataset, Validation dataset
    Output : Validation dataset having predicted labels and probabilities
"""

def predict_model(model, x_val):
    
    try:
        predictions                = model.predict(x_val)
        proba                      = model.predict_proba(x_val)
        probas                     = proba[:,1]
        x_val['predictions']       = predictions
        x_val['probability_score'] = probas
        
        return x_val
   
    except Exception as e:
        logger.error(f"Error occured in Predicting the validation dataset: {e}", exc_info = True)
        pass