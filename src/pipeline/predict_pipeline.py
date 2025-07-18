import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Successfully loaded model and preprocessor")

            print("Transforming input features...")
            data_scaled = preprocessor.transform(features)

            print("Predicting future demand...")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 product_type: str,
                 region: str,
                 past_sales: float,
                 holiday: str,
                 month: int,
                 inventory_level: float):
        self.product_type = product_type
        self.region = region
        self.past_sales = past_sales
        self.holiday = holiday
        self.month = month
        self.inventory_level = inventory_level

    def get_data_as_data_frame(self):
        try:
            input_dict = {
                "product_type": [self.product_type],
                "region": [self.region],
                "past_sales": [self.past_sales],
                "holiday": [self.holiday],
                "month": [self.month],
                "inventory_level": [self.inventory_level]
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)
