import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
            }

            params = {
                "Linear Regression": {},
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20]
                }
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score:.3f}")

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred = best_model.predict(X_test)
            final_r2 = r2_score(y_test, y_pred)
            final_mse = mean_squared_error(y_test, y_pred)
            final_mae = mean_absolute_error(y_test, y_pred)

            logging.info(f"Final R2: {final_r2:.3f}, MSE: {final_mse:.3f}, MAE: {final_mae:.3f}")

            return final_r2
        
        except Exception as e:
            raise CustomException(e, sys)
