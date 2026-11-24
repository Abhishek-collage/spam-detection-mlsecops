import os
from src.data_integrity import record
import pandas as pd
from src.logger import get_logger 
from src.custom_exception import CustomException
from config.config_path import *
from src.custom_function import read_yaml
from dotenv import load_dotenv
import boto3
from sklearn.model_selection import train_test_split
load_dotenv()

logger = get_logger(__name__)
class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        os.makedirs(RAW_DIR, exist_ok=True)

    def download_csv_from_s3(self):
        try:
            client = boto3.client("s3")
            file_path = os.path.join(RAW_FILE_PATH)
            client.download_file(self.bucket_name, self.bucket_file_name, file_path)
            logger.info("Successfully dataset saved")
            
            # Raw file ko bhi manifest mein record kar sakte hain (optional)
            record(file_path)
        except Exception as e:
            logger.error("Error while downloading dataset")
            raise CustomException("Failed to download data from S3", e)

    def split_data(self):
        try:
            logger.info("Starting the splitting process")
            data = pd.read_csv(RAW_FILE_PATH, encoding="latin-1")
            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            
            # --- Yahan train aur test files ko manifest mein record karein ---
            record(TRAIN_FILE_PATH)
            record(TEST_FILE_PATH)
            
            logger.info(f"Train data saved and recorded to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved and recorded to {TEST_FILE_PATH}")
        except Exception as e:
            logger.error("Error while splitting data")
            raise CustomException("Failed to split data into training and test sets", e)

    def run(self):
        try:
            logger.info("Starting data ingestion process ")
            self.download_csv_from_s3()
            logger.info("Data download completed successfully")
            self.split_data()
            logger.info("Data split successfully")
        except CustomException as e:
            logger.error(f"CustomException :{str(e)}")

if __name__ == "__main__":
    config_path = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config_path)
    data_ingestion.run()