
import sys
import uuid
import  os
import pandas as pd

from BigmartsalesPrediction.app_entity.config_entity import DataIngestionConfig, DataValidationConfig, \
    TrainingPipelineConfig, DataTransformationConfig, ModelTrainerConfig, ModelPusherConfig, ModelEvaluationConfig
from BigmartsalesPrediction.app_exception.exception import App_Exception
from BigmartsalesPrediction.app_logger import logging
from BigmartsalesPrediction.app_util.util import read_yaml_file
from BigmartsalesPrediction.constants import *


class Configuration:

    def __init__(self,
                 config_file_path: str = CONFIG_FILE_PATH) -> None:
        try:
            self.config_info = read_yaml_file(file_path=config_file_path)
            self.pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = CURRENT_TIME_STAMP

        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            experiment_id = self.pipeline_config.experiment_id
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]

            train_dataset_download_url = data_ingestion_config_info[DATA_INGESTION_TRAIN_DATA_KEY]
            data_ingestion_dir = data_ingestion_config_info[DATA_INGESTION_DIR_KEY]
            raw_data_dir = data_ingestion_config_info[DATA_INGESTION_RAW_DIR_KEY]
            ingested_data_dir = data_ingestion_config_info[DATA_INGESTION_INGESTED_DIR_KEY]
            raw_file_name = data_ingestion_config_info[DATA_INGESTION_RAW_DATA_FILE_NAME_KEY]
            ingested_train_file_name = data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_FILE_NAME_KEY]
            ingested_test_file_name = data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_FILE_NAME_KEY]

            raw_file_path = os.path.join(artifact_dir, data_ingestion_dir, experiment_id, raw_data_dir, raw_file_name)
            ingested_test_data_path = os.path.join(artifact_dir, data_ingestion_dir, experiment_id, ingested_data_dir,
                                                   ingested_test_file_name)
            ingested_train_data_path = os.path.join(artifact_dir, data_ingestion_dir, experiment_id, ingested_data_dir,
                                                    ingested_train_file_name)
            os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(ingested_test_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(ingested_train_data_path), exist_ok=True)

            data_ingestion_config = DataIngestionConfig(train_dataset_download_url=train_dataset_download_url,
                                                        raw_file_path=raw_file_path,
                                                        ingested_test_data_path=ingested_test_data_path,
                                                        ingested_train_data_path=ingested_train_data_path)

            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            experiment_id = str(uuid.uuid4())
            artifact_dir = os.path.join(ROOT_DIR,
                                        training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            os.makedirs(artifact_dir, exist_ok=True)
            experiment_file_path = os.path.join(artifact_dir, EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            os.makedirs(os.path.dirname(experiment_file_path), exist_ok=True)
            previous_experiment = self.get_previous_experiment(experiment_file_path=experiment_file_path)
            if previous_experiment:
                previous_experiment_id = previous_experiment[0].get('experiment_id')
            else:
                previous_experiment_id = None

            training_pipeline_config = TrainingPipelineConfig(experiment_id=experiment_id,
                                                              previous_experiment_id=previous_experiment_id,
                                                              artifact_dir=artifact_dir,
                                                              experiment_file_path=experiment_file_path)
            logging.info(f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            experiment_id = self.pipeline_config.experiment_id
            previous_experiment_id = self.pipeline_config.previous_experiment_id

            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            data_validation_dir = data_validation_config[DATA_VALIDATION_DIR_KEY]
            data_validation_artifact_dir = os.path.join(
                artifact_dir, data_validation_dir, experiment_id)
            schema_dir = data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY]
            schema_file_name = data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]

            schema_file_path = os.path.join(ROOT_DIR,
                                            schema_dir, schema_file_name
                                            )
            report_dir = data_validation_config[DATA_VALIDATION_REPORT_DIR_KEY]
            report_file_name = data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            report_file_path = os.path.join(data_validation_artifact_dir,
                                            report_dir, report_file_name
                                            )
            os.makedirs(data_validation_artifact_dir, exist_ok=True)
            report_page_file_name = data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            report_page_file_path = os.path.join(data_validation_artifact_dir,
                                                 report_dir, report_page_file_name
                                                 )
            os.makedirs(os.path.dirname(report_file_path), exist_ok=True)

            data_validation_config = DataValidationConfig(experiment_id=experiment_id,
                                                          schema_file_path=schema_file_path,
                                                          previous_experiment_id=previous_experiment_id,
                                                          report_file_path=report_file_path,
                                                          report_page_file_path=report_page_file_path,
                                                          )
            return data_validation_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_previous_experiment(self, experiment_file_path: str) -> str:

        try:
            if os.path.exists(experiment_file_path):
                experiment_pd = pd.read_csv(experiment_file_path)
                experiment_dict = experiment_pd.query('running_status == False').tail(1).to_dict('records')
                return experiment_dict
            else:
                return None
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            experiment_id = self.pipeline_config.experiment_id

            data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            data_transformation_dir = data_transformation_config_info[DATA_TRANSFORMATION_DIR_KEY]
            preprocessed_data_dir = data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]
            preprocessed_file_name = data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            transformed_train_dir_name = data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            transformed_test_dir_name = data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]
            data_transformation_artifact_dir = os.path.join(artifact_dir, data_transformation_dir, experiment_id,
                                                            data_transformation_dir)

            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                preprocessed_data_dir, preprocessed_file_name)

            transformed_train_dir = os.path.join(
                data_transformation_artifact_dir, transformed_train_dir_name)

            transformed_test_dir = os.path.join(
                data_transformation_artifact_dir, transformed_test_dir_name)

            data_transformation_config = DataTransformationConfig(
                preprocessed_object_file_path=preprocessed_object_file_path,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir
            )
            os.makedirs(transformed_test_dir, exist_ok=True)
            os.makedirs(transformed_train_dir, exist_ok=True)
            os.makedirs(os.path.dirname(preprocessed_object_file_path), exist_ok=True)

            logging.info(f"Data transformation config: {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            experiment_id = self.pipeline_config.experiment_id
            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            model_trainer_stack_status = model_trainer_config_info[MODEL_TRAINER_STACKED_KEY]
            model_trainer_artifact_dir_name = model_trainer_config_info[MODEL_TRAINER_ARTIFACT_DIR]
            trained_model_dir_name = model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY]
            trained_model_file_name = model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY]
            model_config_dir = model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY]
            model_config_file_name = model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]

            model_trainer_artifact_dir = os.path.join(
                artifact_dir, model_trainer_artifact_dir_name, experiment_id)

            trained_model_file_path = os.path.join(model_trainer_artifact_dir, trained_model_dir_name,
                                                   trained_model_file_name)

            model_config_file_path = os.path.join(model_config_dir, model_config_file_name)
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)

            base_accuracy = model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]

            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                base_accuracy=base_accuracy,
                stacked=model_trainer_stack_status,
                model_config_file_path=model_config_file_path
            )
            logging.info(f"Model trainer config: {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            artifact_dir = self.pipeline_config.artifact_dir
            experiment_id = self.pipeline_config.experiment_id
            saved_model_dir = os.path.join(ROOT_DIR, model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           )
            model_evaluation_artifact_dir_name = model_evaluation_config[MODEL_EVALUATION_ARTIFACT_DIR]
            model_evaluation_artifact_dir = os.path.join(artifact_dir, model_evaluation_artifact_dir_name)
            model_evaluation_file_name = model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY]
            model_evaluation_file_path = os.path.join(artifact_dir, model_evaluation_artifact_dir,
                                                      model_evaluation_file_name)
            response = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                             time_stamp=self.time_stamp ,saved_model_dir=saved_model_dir)
            os.makedirs(os.path.dirname(model_evaluation_file_path), exist_ok=True)

            logging.info(f"Model Evaluation Config: {response}.")
            return response
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_pusher_config(self) -> ModelPusherConfig:
        try:
            time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            experiment_id = self.pipeline_config.experiment_id
            export_dir_path = os.path.join(ROOT_DIR, model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           time_stamp)

            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)
            logging.info(f"Model pusher config {model_pusher_config}")
            return model_pusher_config

        except Exception as e:
            raise App_Exception(e, sys) from e
