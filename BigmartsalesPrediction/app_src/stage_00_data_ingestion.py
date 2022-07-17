
import sys,os
import pandas as pd

from BigmartsalesPrediction.app_config.configuration import Configuration
from BigmartsalesPrediction.app_entity.config_entity import DataIngestionConfig 
from BigmartsalesPrediction.app_entity.artifacts_entity import DataIngestionArtifact
from BigmartsalesPrediction.app_exception.exception import App_Exception
from BigmartsalesPrediction.app_logger import logging


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise App_Exception(e,sys)
    

    def download_data(self,download_url : str , download_file_path : str) -> str:
        try:
            #extraction remote url to download dataset
            download_df = pd.read_csv(download_url)
            download_df.to_csv(download_file_path, index=False)


            logging.info(f"Downloading file from :[{download_url}] into :[{download_file_path}]")

            return download_file_path

        except Exception as e:
            raise App_Exception(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            train_dataset_download_url = self.data_ingestion_config.train_dataset_download_url
            test_dataset_download_url = self.data_ingestion_config.test_dataset_download_url
            ingested_test_filename = self.data_ingestion_config.ingested_test_filename
            self.download_data(download_url=test_dataset_download_url,download_file_path=ingested_test_filename)
            ingested_train_filename = self.data_ingestion_config.ingested_train_filename
            self.download_data(download_url=train_dataset_download_url,download_file_path=ingested_train_filename)
            
            data_ingestion_response = f"Data Ingestion completed successfully : \n Train dataset : \
                            {ingested_train_filename} \n Test dataset : {ingested_test_filename}"
            logging.info(data_ingestion_response)
            return DataIngestionArtifact(train_file_path= ingested_train_filename  , 
                                         test_file_path = ingested_test_filename,
                                         is_ingested = True, message = data_ingestion_response)
        except Exception as e:
            raise App_Exception(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")

