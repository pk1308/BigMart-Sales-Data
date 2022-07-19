import os
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


ROOT_DIR = os.getcwd()  # to get current working directory
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = get_current_time_stamp()

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

EXPERIMENT_DIR_NAME = "experiment"
EXPERIMENT_FILE_NAME = "experiment.csv"

# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DIR_KEY = "data_ingestion_dir"
DATA_INGESTION_TRAIN_DATA_KEY = 'train_dataset_download_url'
DATA_INGESTION_INGESTED_DIR_KEY = "ingested_dir"
DATA_INGESTION_RAW_DIR_KEY = "raw_data_dir"
DATA_INGESTION_RAW_DATA_FILE_NAME_KEY = "raw_data_file_name"
DATA_INGESTION_INGESTED_TRAIN_FILE_NAME_KEY = "ingested_data_Train_file_name"
DATA_INGESTION_INGESTED_TEST_FILE_NAME_KEY = "ingested_data_Test_file_name"

# Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_ARTIFACT_DIR_NAME = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"
DATA_VALIDATION_SCHEMA_KEY = "columns"
