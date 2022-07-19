from posixpath import split
import sys, os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from BigmartsalesPrediction.app_config.configuration import Configuration
from BigmartsalesPrediction.app_entity.config_entity import DataIngestionConfig
from BigmartsalesPrediction.app_entity.artifacts_entity import DataIngestionArtifact
from BigmartsalesPrediction.app_exception.exception import App_Exception
from BigmartsalesPrediction.app_logger import logging


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 20}Data Ingestion log started.{'<<' * 20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise App_Exception(e, sys)

    def download_data(self, download_url: str, download_file_path: str) -> str:
        try:
            # extraction remote url to download dataset
            download_df = pd.read_csv(download_url)
            download_df.to_csv(download_file_path, index=False)

            logging.info(f"Downloading file from :[{download_url}] into :[{download_file_path}]")

            return download_file_path

        except Exception as e:
            raise App_Exception(e, sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_file_path = self.data_ingestion_config.raw_file_path

            logging.info(f"Reading csv file: [{raw_data_file_path}]")
            raw_data_frame = pd.read_csv(raw_data_file_path)

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index, test_index in split.split(raw_data_frame, raw_data_frame["Outlet_Type"]):
                strat_train_set = raw_data_frame.loc[train_index]
                strat_test_set = raw_data_frame.loc[test_index]

            train_file_path = self.data_ingestion_config.ingested_train_data_path

            test_file_path = self.data_ingestion_config.ingested_test_data_path

            if strat_train_set is not None:
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)

            if strat_test_set is not None:
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data ingestion completed successfully."
                                                            )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.data_ingestion_config
            train_dataset_download_url = data_ingestion_config.train_dataset_download_url
            raw_file_path = data_ingestion_config.raw_file_path
            self.download_data(train_dataset_download_url, raw_file_path)

            data_ingestion_response = self.split_data_as_train_test()
            return data_ingestion_response
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")
