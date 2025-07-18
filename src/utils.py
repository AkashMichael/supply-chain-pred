import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            try:
                print(f"\n🔍 Training model: {model_name}")
                hyperparams = param.get(model_name, {})

                grid = GridSearchCV(model, hyperparams, cv=3, scoring='r2', n_jobs=1)
                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

                # Metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mse = mean_squared_error(y_test, y_pred)

                # Print metrics
                print(f"✅ {model_name} Evaluation:")
                print(f"R2 Score: {r2:.4f}")
                print(f"MAE     : {mae:.4f}")
                print(f"RMSE    : {rmse:.4f}")
                print(f"MSE     : {mse:.4f}")

                report[model_name] = r2  # Use R2 for selecting best model

            except Exception as model_err:
                print(f"❌ Skipping {model_name} due to error: {model_err}")
                continue

        return report

    except Exception as e:
        raise CustomException(e, sys)
