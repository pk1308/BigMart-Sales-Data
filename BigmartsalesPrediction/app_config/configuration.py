
import sys,os
import uuid

from BigmartsalesPrediction.app_entity.config_entity import DataIngestionConfig , TrainingPipelineConfig
from BigmartsalesPrediction.app_util.util import read_yaml_file
from BigmartsalesPrediction.app_logger import logging
from BigmartsalesPrediction.app_exception.exception import App_Exception
from BigmartsalesPrediction.constants import *


class Configuration:

    def __init__(self,
        config_file_path:str =CONFIG_FILE_PATH) -> None:
        try:
            self.config_info  = read_yaml_file(file_path=config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.experiment_id = str(uuid.uuid4())
            self.time_stamp = CURRENT_TIME_STAMP
        except Exception as e:
            raise App_Exception(e,sys) from e


    def get_data_ingestion_config(self) ->DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir=os.path.join(
                artifact_dir,
                self.experiment_id)
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            
            
            test_dataset_download_url=data_ingestion_config_info [TEST_DATA_KEY]
            train_dataset_download_url = data_ingestion_config_info [TRAIN_DATA_KEY]
            
            ingested_dir= data_ingestion_config_info [INGESTED_DIR_KEY]
            ingested_test_dir= os.path.join(data_ingestion_artifact_dir,ingested_dir,INGESTED_TEST_DIR_KEY)
            ingested_train_dir= os.path.join(data_ingestion_artifact_dir,ingested_dir,INGESTED_TRAIN_DIR_KEY)
            os.makedirs(ingested_test_dir,exist_ok=True)
            os.makedirs(ingested_train_dir,exist_ok=True) 
            ingested_test_filename = os.path.join(ingested_test_dir,
                                                  os.path.basename(test_dataset_download_url))
            ingested_train_filename = os.path.join(ingested_train_dir,
                                                   os.path.basename(train_dataset_download_url))           


            data_ingestion_config=DataIngestionConfig(test_dataset_download_url=test_dataset_download_url,
                                          train_dataset_download_url = train_dataset_download_url,
                                          ingested_test_filename= ingested_test_filename,
                                           ingested_train_filename =  ingested_train_filename )
            
            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise App_Exception(e,sys) from e
        
    def get_training_pipeline_config(self) ->TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR,
            training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
            training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            )

            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training pipleine config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise App_Exception(e,sys) from e
