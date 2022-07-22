import os
import sys

from BigmartsalesPrediction.app_config.configuration import Configuration
from BigmartsalesPrediction.app_pipeline.pipeline import Pipeline
from BigmartsalesPrediction.app_exception.exception import App_Exception
from BigmartsalesPrediction.app_logger import logging


def main():
    try:
        config_path = os.path.join("config", "config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_path))
        # pipeline.run_pipeline()
        pipeline.start()
        logging.info("main function execution completed.")

    except Exception as e:
        logging.error(f"{e}")
        raise App_Exception(e, sys)


if __name__ == "__main__":
    main()
