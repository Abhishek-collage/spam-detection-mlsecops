import os
from config.config_path import (
    CONFIG_PATH,
    PREPROCESS_DIR,
    PREPROCESS_TEST_DATA_PATH,
    PREPROCESS_TRAIN_DATA_PATH,
    TEST_FILE_PATH,
    TRAIN_FILE_PATH,
)
from src.custom_exception import CustomException
from src.custom_function import read_yaml
from src.data_integrity import load_verified
from src.data_validation import validate
from src.logger import get_logger
from src.data_integrity import load_verified, record

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        os.makedirs(self.processed_dir, exist_ok=True)

    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            record(file_path, stage="preprocessed")
            logger.info(f"Data saved and recorded successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error during saving data step: {e}")
            raise CustomException("Error while saving data", e) from e

    def preprocess_data(self):
        try:
            logger.info("Data preprocessing starting")
            prep_config = self.config["data_preprocessing"]
            target_column = prep_config["target_column"]
            mapping = prep_config["label_mapping"]

            if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
                raise FileNotFoundError(
                    f"Train or test file missing. Train: {self.train_path}, Test: {self.test_path}"
                )

            logger.info("Loading verified train and test data")
            train_df = load_verified(self.train_path)
            test_df = load_verified(self.test_path)

            logger.info("Validating train and test schemas with Pandera")
            train_df = validate(train_df)
            test_df = validate(test_df)

            train_df = train_df.dropna(subset=[target_column]).copy()
            test_df = test_df.dropna(subset=[target_column]).copy()

            mapped_train = train_df[target_column].map(mapping)
            mapped_test = test_df[target_column].map(mapping)

            if mapped_train.isna().any() or mapped_test.isna().any():
                raise ValueError(f"Unmapped label values found in column '{target_column}'")

            train_df[target_column] = mapped_train.astype(int)
            test_df[target_column] = mapped_test.astype(int)

            self.save_data(train_df, PREPROCESS_TRAIN_DATA_PATH)
            self.save_data(test_df, PREPROCESS_TEST_DATA_PATH)
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline: {e}")
            raise CustomException("Error while data processing pipeline", e) from e

    def process(self):
        try:
            self.preprocess_data()
            logger.info("Data Preprocessing pipeline finished")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise


if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESS_DIR, CONFIG_PATH)
    processor.process()