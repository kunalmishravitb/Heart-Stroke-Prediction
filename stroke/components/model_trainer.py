import sys
import numpy as np
import pandas as pd

from typing import List, Tuple
from stroke.entity.artifact_entity import (ClassificationMetricArtifact,
                                           DataTransformationArtifact,
                                           ModelTrainerArtifact)
from stroke.entity.config_entity import ModelTrainerConfig
from stroke.ml.estimator import HeartStrokeModel
from stroke.exception import HeartStrokeException
from stroke.logger import logging
from stroke.utils.main_utils import (load_numpy_array_data, load_object,
                                     read_yaml_file, save_object)
from neuro_mf import ModelFactory # neuro_mf means Neuro Model Factory. It will make use of model.yaml file to train the model. ModelFactory is responsible for fine tuning.
from pandas import DataFrame
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    
    # This is used to load the model object and get the classification report out of it
    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name : get_model_object_and_report
        Description : This function uses neuro_mf to get the best model object and report of the best model

        Output      : Returns metric artifact object and best model object
        On Failure  : Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path = self.model_trainer_config.model_config_file_path)
            # Splitting the train and test
            x_train, y_train, x_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1]
            )
            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            # Give the best model which neuro-mf has created
            model_obj = best_model_detail.best_model
            y_pred = model_obj.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            return best_model_detail, metric_artifact
        except Exception as e:
            raise HeartStrokeException(e, sys) from e
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name : initiate_model_trainer
        Description : This function initiates a model trainer steps

        Output      : Returns model trainer artifact
        On Failure  : Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            heart_stroke_model = HeartStrokeModel(preprocessing_object=preprocessing_obj,
                                                  trained_model_object=best_model_detail.best_model)
            logging.info("Created Heart Stroke object with preprocessor and model")
            logging.info("Created best model file path.")
            # Save the object in trained_model_file_path which we have taken from the configuration
            save_object(self.model_trainer_config.trained_model_file_path, heart_stroke_model)

            # This information will be passed to the model evaluation component
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise HeartStrokeException(e, sys) from e
