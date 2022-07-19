from collections import namedtuple


Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp",
                                       "artifact_dir", "running_status", "start_time",
                                       "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])
