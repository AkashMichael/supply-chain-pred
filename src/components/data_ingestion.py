import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")
        try:
            # ✅ Correct and verified file path
            data_path = 'notebook/data/Retail-Supply-Chain-Sales-Dataset.xlsx'

            # ✅ Check if file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"File not found: {data_path}")

            # ✅ Load the Excel file
            df = pd.read_excel(data_path)
            logging.info("Dataset loaded successfully.")

            # ✅ Create artifacts folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # ✅ Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # ✅ Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed.")

            # ✅ Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test data saved.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in data ingestion.")
            raise CustomException(e, sys)

# ✅ Optional direct test
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"✅ Data Ingestion Complete.\nTrain path: {train_path}\nTest path: {test_path}")
