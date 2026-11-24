import os
import sys

from config.config_path import (
    CONFIG_PATH,
    MODEL_OUTPUT_PATH,
    PREPROCESS_DIR,
    PREPROCESS_TEST_DATA_PATH,
    PREPROCESS_TRAIN_DATA_PATH,
    TEST_FILE_PATH,
    TRAIN_FILE_PATH,
)
from src.custom_exception import CustomException
from src.custom_function import read_yaml
from src.data_ingestion import DataIngestion
from src.data_integrity import verify_all
from src.data_preprocessing import DataProcessor
from src.logger import get_logger
from src.model_training import ModelTraining
from src.drift import run_drift_check
logger = get_logger(__name__)


def run_pipeline(verify_manifest: bool = True, run_drift: bool = False, current_data_path: str | None = None):
    try:
        logger.info("Starting MLSecOps Spam Classification pipeline")

        # 1. Config & Data Ingestion
        config_data = read_yaml(CONFIG_PATH)
        ingestion = DataIngestion(config_data)
        ingestion.run()

        # 2. Manifest Integrity Check (ingested/split files verified)
        if verify_manifest:
            logger.info("Verifying data integrity manifest")
            verify_all()

        # 3. Data Validation & Preprocessing
        processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESS_DIR, CONFIG_PATH)
        processor.process()

        # 4. Model Training & MLflow Tracking
        trainer = ModelTraining(
            PREPROCESS_TRAIN_DATA_PATH,
            PREPROCESS_TEST_DATA_PATH,
            MODEL_OUTPUT_PATH,
        )
        trainer.run()

        # 5. Optional Drift Check on new/current data
        if run_drift:
            if not current_data_path:
                raise ValueError("--current-data path required when --drift is enabled")
         

            run_drift_check(current_data_path)

        logger.info("Pipeline completed successfully")
    except CustomException as e:
        logger.error(f"Pipeline failed with CustomException: {e}")
        raise
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        raise CustomException("Pipeline execution failed", e) from e


if __name__ == "__main__":
    drift_flag = "--drift" in sys.argv
    curr_path = None
    if "--current-data" in sys.argv:
        idx = sys.argv.index("--current-data")
        if idx + 1 < len(sys.argv):
            curr_path = sys.argv[idx + 1]

    run_pipeline(
        verify_manifest="--no-verify" not in sys.argv,
        run_drift=drift_flag,
        current_data_path=curr_path,
    )