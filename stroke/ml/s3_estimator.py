# Here we will be writing the code which will be used to retrieve the model from s3 bucket and this can be used locally for prediction

import os, sys

from pandas import DataFrame
from stroke.cloud_storage.aws_storage import SimpleStorageService
from stroke.ml.estimator import HeartStrokeModel
from stroke.exception import HeartStrokeException


class StrokeEstimator:
    """
    This class is used to save and retrieve heart_stroke model in s3 bucket and to do prediction
    """
    def __init__(self, bucket_name, model_path):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: HeartStrokeModel = None
    

    def is_model_present(self, model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path) # we have model_path called as model registry
        except HeartStrokeException as e:
            print(e)
            return False
    

    def load_model(self) -> HeartStrokeModel:
        """
        Load the model from the model_path
        :return:
        """
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
    

    def save_model(self, from_file, remove:bool=False) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that means you will have your model locally available in your system folder
        :return:
        """
        try:
            self.s3.upload_file(from_file, # This will be the path to the respective pickle file
                                to_filename=self.model_path, # This is the model path which is already available as a registry key
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            raise HeartStrokeException(e, sys)
    

    def predict(self, dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model=self.load_model() # It will load the model and once the model is loaded it will automatically take dataframe as a final predict function
            return self.loaded_model.predict(dataframe=dataframe) # Here it will predict it and their prediction will be returned as an output
        except Exception as e:
            raise HeartStrokeException(e, sys)
