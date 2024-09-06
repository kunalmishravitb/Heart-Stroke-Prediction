import yaml, sys, shutil
import numpy as np
import os.path
import dill
import pandas as pd
from typing import Dict, Tuple
from pandas import DataFrame
from yaml import safe_dump
# from stroke.constant.training_pipeline import(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH, SCHEMA_FILE_PATH)
from stroke.exception import HeartStrokeException
from stroke.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HeartStrokeException(e, sys) from e


def write_yaml_file(file_path:str, content: object, replace: bool=False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise HeartStrokeException(e, sys)


# Loading the object (pickle file, class file, etc) into the variable
def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of MainUtils class")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info("Exited the load_object method of MainUtils class")
        return obj
    except Exception as e:
        raise HeartStrokeException(e, sys) from e


# This is the function used to save the numpy into npy format
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HeartStrokeException(e, sys) from e


# Function used to load the numpy array
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HeartStrokeException(e, sys) from e
    

# Function used to save any kind of object like pickle, class, etc
def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of MainUtils class")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise HeartStrokeException(e, sys) from e
    


# def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
#     try:
#         dataset_schema = read_yaml_file(schema_file_path)
#         schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]
#         dataframe = pd.read_csv(file_path)
#         error_message = ""
#         for column in dataframe.columns:
#             if column in list(schema.keys()):
#                 dataframe[column].astype(schema[column])
#             else:
#                 error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
#         if len(error_message) > 0:
#             raise Exception(error_message)
#         return dataframe
#     except Exception as e:
#         raise HeartStrokeException(e, sys) from e
