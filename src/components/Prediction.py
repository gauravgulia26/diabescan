import os
import joblib
import numpy as np
from src.logger.custom_logger import logger
from src.constants import MODEL_DIR_PATH
import warnings
warnings.filterwarnings('ignore')

class SVMPredictor:
    def __init__(self, model_path=MODEL_DIR_PATH):
        """
        Initializes the SVMPredictor by loading the saved model.

        Parameters:
        - model_path (str): Path to the saved model file.
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the saved SVM model from the specified path.

        Returns:
        - Loaded model object.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found.")
        try:
            model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from '{self.model_path}'.")
            return model
        except Exception as e:
            raise IOError(f"An error occurred while loading the model: {e}")

    def predict(self, input_data):
        """
        Predicts the class labels for the given input data.

        Parameters:
        - input_data (list or numpy.ndarray): Input data with shape (n_samples, 16).

        Returns:
        - predictions (numpy.ndarray): Predicted class labels.
        """
        # Convert input_data to numpy array if it's a list
        if isinstance(input_data, list):
            input_data = np.array(input_data)

        # Validate input shape
        # if input_data.ndim != 2 or input_data.shape[1] != 16:
        #     raise ValueError(f"Expected input data with shape (n_samples, 16), got {input_data.shape}.")

        try:
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            raise RuntimeError(f"An error occurred during prediction: {e}")
        

if __name__ == "__main__":
    # Instantiate the predictor
    predictor = SVMPredictor(model_path=MODEL_DIR_PATH)

    # Example input data with 16 features
    sample_input = [
        [0.5, 1.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8,
         9.9, 10.1, 11.2, 12.3, 13.4, 14.5, 15.6, 16.7,0]
    ]

    # Make prediction
    try:
        prediction = predictor.predict(sample_input)
        logger.info(f"Predicted class: {prediction}")
        if int(prediction) == 1:
            print('You have Diabetes.')
        else:
            print('You are Healthy.')
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
