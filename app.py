from cmath import log
from flask import Flask, request
import sys
import os

from BigmartsalesPrediction.app_util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from BigmartsalesPrediction.app_logger import logging , get_log_dataframe
from BigmartsalesPrediction.app_exception.exception import App_Exception
import json
from BigmartsalesPrediction.app_config.configuration import Configuration
from BigmartsalesPrediction.constants import CONFIG_DIR, get_current_time_stamp
from BigmartsalesPrediction.app_pipeline.pipeline import Pipeline
from BigmartsalesPrediction.app_entity.app_predictor import Prediction_Data, App_predictor
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "Bigmartsales"
SAVED_MODELS_DIR_NAME = "production_model"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

PREDICTOR_DATA_KEY = "predictor_data"
ITEM_OUTLET_SALES = "item_outlet_sales"


app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'BigmartsalesPrediction'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("BigmartsalesPrediction", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration())
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
 
    context = {
        PREDICTOR_DATA_KEY: None,
        ITEM_OUTLET_SALES : None
    }
    logging.info("predict called ")

    if request.method == 'POST':
        try :



            Item_Identifier = request.form['Item_Identifier']
            Item_Fat_Content = request.form['Item_Fat_Content']
            Item_Type = request.form['Item_Type']
            Outlet_Identifier = request.form['Outlet_Identifier']
            Outlet_Type = request.form['Outlet_Type']
            
            Item_MRP = float(request.form['Item_MRP'])
            Item_Visibility = float(request.form['Item_Visibility'])
            Item_Weight = float(request.form['Item_Weight'])
            Outlet_Establishment_Year= int(request.form['Outlet_Establishment_Year'])
            Outlet_Location_Type = request.form['Outlet_Location_Type']
            Outlet_Size = request.form['Outlet_Size']

            prediction_data = Prediction_Data(Item_Identifier=Item_Identifier,
                                              Item_Fat_Content = Item_Fat_Content,
                                              Item_Type = Item_Type,
                                              Outlet_Identifier = Outlet_Identifier,
                                              Outlet_Type = Outlet_Type,
                                              Item_MRP = Item_MRP,
                                            Item_Visibility = Item_Visibility,
                                            Item_Weight = Item_Weight,
                                            Outlet_Establishment_Year = Outlet_Establishment_Year , 
                                            Outlet_Location_Type = Outlet_Location_Type,
                                            Outlet_Size = Outlet_Size)
                                            
            logging.info(f"prediction_data: {prediction_data}")
            prediction_df = prediction_data.get_housing_input_data_frame()
            
            app_predictor = App_predictor(model_dir=MODEL_DIR)
            logging.info(f"app_predictor: {app_predictor}")
            Item_Outlet_Sales= app_predictor.predict(X=prediction_df)
            context = {     
                PREDICTOR_DATA_KEY : prediction_data.get_housing_data_as_dict(),
                ITEM_OUTLET_SALES: Item_Outlet_Sales,
            }
            logging.info(f"context: {context}")
        except Exception as e:
            raise App_Exception(e , sys)
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()
