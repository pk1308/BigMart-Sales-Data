from loguru import logger as logging
from datetime import datetime
import os
import pandas as pd
from BigmartsalesPrediction.constants import get_current_time_stamp

LOG_DIR = "logs"


def get_log_file_name():
    return f"log_{get_current_time_stamp()}.log"


LOG_FILE_NAME = get_log_file_name()

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
log_format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} |{function} |\n {message}"
logging.add(LOG_FILE_PATH, format=log_format, level="DEBUG")


def get_log_dataframe(file_path):
    data = []
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("|"))

    log_df = pd.DataFrame(data)
    columns = ["Time stamp", "Log Level", "function name", "message"]
    log_df.columns = columns

    log_df["log_message"] = log_df['Time stamp'].astype(str) + ":$" + log_df["message"]

    return log_df[["log_message"]]
