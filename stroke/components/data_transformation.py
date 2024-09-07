import sys
from typing import Union
import numpy as np
import pandas as pd

from stroke.constant.training_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH
from stroke.entity.artifact_entity import (DataIngestionArtifact,
                                           DataTransformationArtifact, DataValidationArtifact)
from stroke.entity.config_entity import DataTransformationConfig
from stroke.utils.main_utils import read_yaml_file
from stroke.exception import HeartStrokeException
from stroke.logger import logging
from stroke.utils.main_utils import save_numpy_array_data, save_object
from imblearn.combine import SMOTEENN # SMOTEENN will handle imbalanced data and it will give new train and test data
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH) # This will get the information of all the different columns like categorical, transformation columns, etc
        except Exception as e:
            raise HeartStrokeException(e, sys)
    

    # This is used to read the data from a given path
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HeartStrokeException(e, sys)
    

    # It will create column transformer object and it will return it. Here we will take information from schema.yaml file
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name : get_data_transformer_object
        Description : This method creates and returns a data transformer object

        Output      : data transformer object is created and returned
        On Failure  : Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )
        try:
            logging.info("Got numerical, categorical, transformation columns from schema config")
            # Reading the schema config
            numerical_columns = self._schema_config['Numerical_columns']
            categorical_columns = self._schema_config['Categorical_columns']
            transform_columns = self._schema_config['Transformation_columns']

            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            transform_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer(standardize=True)) # We have skewed data i.e. avg_glucose_level so we are using power transformer so that it will be well-shaped curve
            ])

            preprocessor = ColumnTransformer(
                [
                    ("Numeric_Pipeline", numeric_pipeline, numerical_columns),
                    ("Categorical_Pipeline", categorical_pipeline, categorical_columns),
                    ("Power_Transformation", transform_pipe, transform_columns)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return preprocessor
        except Exception as e:
            raise HeartStrokeException(e, sys) from e
    

    # DataTransformationArtifact will have information of the path of train and test npy and also the path of preprocessing.pickle file
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name : initiate_data_transformation
        Description : This method initiates the data transformation component for the pipeline

        Output      : data transformer steps are performed and preprocessor object is created
        On Failure  : Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status: # Checking if the validation status is true or false
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(colummns=[TARGET_COLUMN], axis=1) # X_train
                target_feature_train_df = train_df[TARGET_COLUMN] # Y_train
                logging.info("Got train features and test features of Training dataset")
                
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1) # X_test
                target_feature_test_df = test_df[TARGET_COLUMN] # Y_test
                logging.info("Got train features and test features of Testing dataset")
                
                logging.info("Applying preprocessing object on training dataframe and testing dataframe")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Used the preprocessor object to fit transform the train features")
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Used the preprocessor object to transform the test features")
                
                logging.info("Applying SMOTEENN on Training dataset")
                smt = SMOTEENN(sampling_strategy="minority")
                # This step will only resample the train data
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                logging.info("Applied SMOTEENN on training dataset")
                
                logging.info("Applying SMOTEENN on testing dataset")
                # This step will only resample the test data
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )
                logging.info("Applied SMOTEENN on testing dataset")
                
                logging.info("Created train array and test array")
                # Concatenation of input and target columns
                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]
                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                # Save the object of preprocessing pickle file in the defined path
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
                logging.info("Saved the preprocessor object")
                logging.info("Exited initiate_data_transformation method of Data_Transformation class")

                # All this information will be saved in the artifact
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path, # This is the object path where we save the pickle file
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path, # This is the object path where we save the train file path
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path # This is the object path where we save the test file path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            raise HeartStrokeException(e, sys) from e
