import os
from re import S
import sys

from BigmartsalesPrediction.app_exception.exception import App_Exception
from BigmartsalesPrediction.app_util.util import load_object

import pandas as pd


class Prediction_Data:

    def __init__(self, Item_Identifier: str,
                 Item_Fat_Content: str,
                 Item_Type: str,
                 Outlet_Identifier: str,
                 Outlet_Type: str,
                 Item_MRP: float,
                 Item_Visibility: float,
                 Item_Weight: float,
                 Outlet_Establishment_Year: int,
                 Outlet_Location_Type :str ,
                Outlet_Size: str,
                 Item_Outlet_Sales: float = None):

        try:
            self.Item_Identifier = Item_Identifier
            self.Item_Fat_Content = Item_Fat_Content
            self.Item_Type = Item_Type
            self.Outlet_Identifier = Outlet_Identifier
            self.Outlet_Type = Outlet_Type
            self.Item_MRP = Item_MRP
            self.Item_Visibility = Item_Visibility
            self.Item_Weight = Item_Weight
            self.Outlet_Establishment_Year = Outlet_Establishment_Year
            self.Outlet_Location_Type = Outlet_Location_Type
            self.Outlet_Size = Outlet_Size
            self.Item_Outlet_Sales = Item_Outlet_Sales
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_housing_input_data_frame(self):

        try:
            housing_input_dict = self.get_housing_data_as_dict()
            return pd.DataFrame(housing_input_dict)
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_housing_data_as_dict(self):
        try:
            input_data = {
                "Item_Identifier": [self.Item_Identifier],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Type": [self.Item_Type],
                "Outlet_Identifier": [self.Outlet_Identifier],
                "Outlet_Type": [self.Outlet_Type],
                "Item_MRP": [self.Item_MRP],
                "Item_Visibility": [self.Item_Visibility],
                "Item_Weight": [self.Item_Weight],
                'Outlet_Location_Type': [self.Outlet_Location_Type],
                'Outlet_Size': [self.Outlet_Size],
                "Outlet_Establishment_Year": [self.Outlet_Establishment_Year]
            }

            return input_data
        except Exception as e:
            raise App_Exception(e, sys)


class App_predictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise App_Exception(e, sys) from e

    def predict(self, X):
        try:
            model_path = os.path.join(self.model_dir , "model.pkl")
            model = load_object(file_path=model_path)
            median_house_value = model.predict(X)
            return median_house_value
        except Exception as e:
            raise App_Exception(e, sys) from e
