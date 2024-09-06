import os
from pathlib import Path # Prevent the file path error like forward slash in linux and backward slash in windows
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

#project_name="mlProject"

list_of_files=[
    f"stroke/__init__.py",
    f"stroke/cloud_storage/__init__.py",
    f"stroke/components/__init__.py",
    f"stroke/utils/__init__.py",
    f"stroke/utils/main_utils.py",
    f"stroke/ml/__init__.py",
    f"stroke/data_access/__init__.py",
    f"stroke/data_access/heart_stroke_data.py",
    f"stroke/configuration/__init__.py",
    f"stroke/configuration/aws_connection.py",
    f"stroke/configuration/mongo_db_connection.py",
    f"stroke/logger/__init__.py",
    f"stroke/entity/__init__.py",
    f"stroke/exception/__init__.py",
    f"stroke/pipeline/__init__.py",
    f"stroke/constant/__init__.py",
    f"stroke/constant/application.py",
    f"stroke/constant/database.py",
    f"stroke/constant/env_variable.py",
    f"stroke/constant/s3_bucket.py",
    "config/model.yaml",
    "config/schema.yaml",
    # "params.yaml",
    # "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
    "templates/experiment_history.html",
    "templates/files.html",
    "templates/header.html",
    "templates/log.html",
    "templates/log_files.html",
    "templates/predict.html",
    "templates/train.html",
    "templates/saved_models_files.html",
    "templates/update_model.html"
]

for filepath in list_of_files:
    filepath=Path(filepath) # Based on OS it will automatically detect and convert the slash into particular OS slash and prevent the error
    filedir,filename=os.path.split(filepath) # Separating the folder and file

    # Creating file directory
    if filedir!="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")
    
    # Creating file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): # it will create the file if it does not exist or the size of the file is zero
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")
