import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def main():
    try:
        logging.info("🔁 Starting model training pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )

        # Step 3: Model Training
        trainer = ModelTrainer()
        final_r2_score = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"✅ Pipeline completed. Final R2 score: {final_r2_score:.4f}")
        print(f"\n📈 Final model R2 score on test set: {final_r2_score:.4f}")

    except Exception as e:
        logging.error("❌ Pipeline execution failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
