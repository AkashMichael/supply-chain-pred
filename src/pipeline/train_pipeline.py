import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def main():
    try:
        logging.info("🚀 Starting demand forecasting training pipeline...")


        # Check if file exists
       
        # Load dataset
        df = pd.read_excel("notebook/data/preprocessed_dataset.xlsx")

        print("📊 Columns in dataset:", df.columns.tolist())


        # ✅ Required features and target
        features = ["lag_1", "lag_7", "lag_30", "rolling_mean_7", "month", "day_of_week"]
        target = "demand"

        if not all(col in df.columns for col in features + [target]):
            raise ValueError("❌ Missing required columns in dataset.")

        X = df[features]
        y = df[target]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing pipeline
        preprocessor = Pipeline([
            ('scaler', StandardScaler())
        ])

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        # Save preprocessor
        save_object("artifacts/preprocessor.pkl", preprocessor)

        # Models to evaluate
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        best_model = None
        best_rmse = float("inf")

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5  # Manually take square root
 
            logging.info(f"📈 {name} RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse

        # Save best model
        save_object("artifacts/model.pkl", best_model)

        logging.info(f"✅ Training complete. Best RMSE: {best_rmse:.4f}")
        print(f"\n🎯 Best Model RMSE: {best_rmse:.4f}")

    except Exception as e:
        logging.error("❌ Training pipeline failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
