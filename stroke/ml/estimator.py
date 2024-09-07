# This file is basically used for saving the model as a pickle file

import os
import sys

from dataclasses import dataclass
from pandas import DataFrame
from stroke.exception import HeartStrokeException
from stroke.logger import logging
from sklearn.compose import ColumnTransformer


class HeartStrokeModel:
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocessor
        :param trained_model_object: Input Object of trained model
        """
        # Initializing it in the constructor
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object


        def predict(self, dataframe: DataFrame) -> DataFrame:
            """
            Function accepts raw inputs and then transformed raw input using preprocessing_object
            which guarantees that the inputs are in the same format as the training data
            At last it performs prediction on transformed features
            """
            logging.info("Entered predict method of HeartStrokeModel class")
            try:
                logging.info("Using the trained model to get predictions")
                # Here we are transforming the data before predicting it
                transformed_feature = self.preprocessing_object.transform(dataframe)
                logging.info("Used the trained model to get predictions")
                return self.trained_model_object.predict(transformed_feature) # transformed_feature will be the prediction for our model
            except Exception as e:
                raise HeartStrokeException(e, sys) from e
        

        def __repr__(self):
            return f"{type(self.trained_model_object).__name__}()"
        

        def __str__(self):
            return f"{type(self.trained_model_object).__name__}()"
