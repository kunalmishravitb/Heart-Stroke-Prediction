from email import header
import sys, os
import numpy as np
from pandas import DataFrame
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from stroke.entity.config_entity import DataIngestionConfig
from stroke.entity.artifact_entity import DataIngestionArtifact
from stroke.exception import HeartStrokeException
from stroke.logger import logging
from stroke.utils.main_utils import read_yaml_file
from stroke.data_access.heart_stroke_data import StrokeData
from stroke.constant.training_pipeline import SCHEMA_FILE_PATH
from stroke.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HeartStrokeException(e, sys)
    

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name: export_data_into_feature_store
        Description: This method exports the dataframe from mongodb feature store as dataframe

        Output: DataFrame
        On Failure: Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            heart_stroke_data = StrokeData()
            dataframe = heart_stroke_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path # This is the path where our data will be stored
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(
                f"Saving exported data into feature store file path: {feature_store_file_path}"
            )
            dataframe.to_csv(feature_store_file_path, index=False, header=True) # Saving the dataframe into csv
            return dataframe
        except Exception as e:
            raise HeartStrokeException(e, sys)
    

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name: split_data_as_train_test
        Description: This method splits the dataframe into train set and test set based on split ratio

        Output: Folder is created in s3 bucket
        On Failure: Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        
        try:
            # Split data into train and test
            train_set, test_set = train_test_split(
                dataframe, test_size = self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise HeartStrokeException(e, sys) from e
    

    # This function will return data ingestion artifact
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: This method initiates the data ingestion components of training pipeline

        Output: train set and test set are returned as the artifacts of data ingestion components
        On Failure: Write an exception log and then raise an exception
        """
        
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            dataframe = self.export_data_into_feature_store() # This will return the dataframe
            _schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH) # It will check the drop column in schema.yaml file
            dataframe = dataframe.drop(_schema_config["Drop_columns"], axis=1) # This will drop the columns which is mention inside schema.yaml file
            logging.info("Got the data from mongodb")
            self.split_data_as_train_test(dataframe) # Calling the method to split the data into train and test
            logging.info("Performed train test split on the dataset")
            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
            # Giving trained and test file path in which we have saved the csv. So this path information will be passed to the next components
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise HeartStrokeException(e, sys) from e
