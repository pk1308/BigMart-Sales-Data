from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["train_dataset_download_url", "raw_file_path", "ingested_test_data_path",
                                  'ingested_train_data_path'])

DataValidationConfig = namedtuple("DataValidationConfig",
                                  ["experiment_id", "previous_experiment_id", "schema_file_path", "report_file_path",
                                   "report_page_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_dir",
                                                                   "transformed_test_dir",
                                                                   "preprocessed_object_file_path"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig",
                                ["trained_model_file_path", "base_accuracy", "model_config_file_path",
                                 "stacked"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_file_path", 
                                                             "saved_model_dir" , "time_stamp"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",
                                    ["experiment_id", "previous_experiment_id", "artifact_dir", "experiment_file_path"])
