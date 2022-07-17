from threading import Thread
import os , sys 
from datetime import datetime
import pandas as pd

from BigmartsalesPrediction.app_config.configuration import Configuration
from BigmartsalesPrediction.app_exception.exception import App_Exception
from BigmartsalesPrediction.app_logger import logging
from BigmartsalesPrediction.constants import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME
from BigmartsalesPrediction.app_entity.experiment_entity import Experiment
from BigmartsalesPrediction.app_entity.config_entity import DataIngestionConfig
from BigmartsalesPrediction.app_entity.artifacts_entity import DataIngestionArtifact
from BigmartsalesPrediction.app_src.stage_00_data_ingestion import DataIngestion





class Pipeline(Thread):
    experiment: Experiment = Experiment(*([None] * 11))
    experiment_file_path = None

    def __init__(self, config: Configuration ) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False, name="pipeline")
            self.config = config
            self.experiment_id = config.experiment_id
        except Exception as e:
            raise App_Exception(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise App_Exception(e, sys) from e
    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            # data ingestion
            logging.info("Pipeline starting.")


            Pipeline.experiment = Experiment(experiment_id=self.experiment_id,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_time_stamp=self.config.time_stamp,
                                             running_status=True,
                                             start_time=datetime.now(),
                                             stop_time=None,
                                             execution_time=None,
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             is_model_accepted=None,
                                             message="Pipeline has been started.",
                                             accuracy=None,
                                             )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            self.save_experiment()

            data_ingestion_artifact = self.start_data_ingestion()
            # data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # data_transformation_artifact = self.start_data_transformation(
            #     data_ingestion_artifact=data_ingestion_artifact,
            #     data_validation_artifact=data_validation_artifact
            # )
            # model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            # model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
            #                                                         data_validation_artifact=data_validation_artifact,
            #                                                         model_trainer_artifact=model_trainer_artifact)

            # if model_evaluation_artifact.is_model_accepted:
            #     model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
            #     logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            # else:
            #     logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")

            # stop_time = datetime.now()
            # Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,
            #                                  initialization_timestamp=self.config.time_stamp,
            #                                  artifact_time_stamp=self.config.time_stamp,
            #                                  running_status=False,
            #                                  start_time=Pipeline.experiment.start_time,
            #                                  stop_time=stop_time,
            #                                  execution_time=stop_time - Pipeline.experiment.start_time,
            #                                  message="Pipeline has been completed.",
            #                                  experiment_file_path=Pipeline.experiment_file_path,
            #                                  is_model_accepted=model_evaluation_artifact.is_model_accepted,
            #                                  accuracy=model_trainer_artifact.model_accuracy
            #                                  )
            # logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
        except Exception as e:
            raise App_Exception(e, sys) from e

    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise e

    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(Pipeline.experiment.experiment_file_path)]})

                experiment_report = pd.DataFrame(experiment_dict)

                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path, mode="w", index=False, header=True)
            else:
                print("First start experiment")
        except Exception as e:
            raise App_Exception(e, sys) from e

    @classmethod
    def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit = -1 * int(limit)
                return df[limit:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise App_Exception(e, sys) from e

