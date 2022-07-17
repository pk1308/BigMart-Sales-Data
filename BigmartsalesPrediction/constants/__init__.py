import os
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    
ROOT_DIR = os.getcwd()  #to get current working directory
CONFIG_DIR =  os.path.join(ROOT_DIR, 'config')
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR,CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = get_current_time_stamp()

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

EXPERIMENT_DIR_NAME="experiment"
EXPERIMENT_FILE_NAME="experiment.csv"

# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
TEST_DATA_KEY = 'test_dataset_download_url'
TRAIN_DATA_KEY = 'train_dataset_download_url'
INGESTED_DIR_KEY= "ingested_dir" 
INGESTED_TRAIN_DIR_KEY = 'ingested_train'
INGESTED_TEST_DIR_KEY = 'ingested_test'