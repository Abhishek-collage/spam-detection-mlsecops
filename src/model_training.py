import os
import joblib
import mlflow
import dagshub
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from config.config_path import MODEL_OUTPUT_PATH, PREPROCESS_TEST_DATA_PATH, PREPROCESS_TRAIN_DATA_PATH
from src.custom_exception import CustomException
from src.data_integrity import load_verified
from src.logger import get_logger

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
logger = get_logger(__name__)
load_dotenv()

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

    def load_and_split_data(self):
        try:
            logger.info(f"Loading verified train data from {self.train_path}")
            train_df = load_verified(self.train_path)
            logger.info(f"Loading verified test data from {self.test_path}")
            test_df = load_verified(self.test_path)

            X_train = train_df["Message"].fillna("")
            y_train = train_df["Category"].astype(int)

            X_test = test_df["Message"].fillna("")
            y_test = test_df["Category"].astype(int)

            logger.info("Data loaded and split successfully for model training")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e) from e

    def model_training(self, X_train, Y_train):
        try:
            logger.info("Initializing model pipeline")
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)),
                ("clf", LogisticRegression(max_iter=1000)),
            ])
            pipeline.fit(X_train, Y_train)
            logger.info("Model training completed successfully")
            return pipeline
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed model training", e) from e

    def evaluate_model(self, model, X_test, Y_test):
        try:
            logger.info("Evaluating model")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred, zero_division=0)
            recall = recall_score(Y_test, y_pred, zero_division=0)
            f1 = f1_score(Y_test, y_pred, zero_division=0)

            logger.info(
                f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | F1: {f1:.4f}"
            )

            return {
                "recall_score": recall,
                "precision_score": precision,
                "accuracy_score": accuracy,
                "f1_score": f1,
            }
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e) from e

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info("Saving the model locally")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved locally to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException("Failed to save model", e) from e

    def run(self):
        try:
            dagshub.auth.add_app_token(os.environ["MLFLOW_TRACKING_PASSWORD"])
            dagshub.init(
                repo_owner="abhishekla314",
                repo_name="spam-detection-mlsecops",
                mlflow=True,
            )
            
            mlflow.set_experiment("Spam_Classification")
            with mlflow.start_run():
                logger.info("Starting model training pipeline")

                X_train, y_train, X_test, y_test = self.load_and_split_data()

                logger.info("Logging datasets to mlflow")
                if os.path.exists(self.test_path):
                    mlflow.log_artifact(self.test_path, artifact_path="datasets")
                if os.path.exists(self.train_path):
                    mlflow.log_artifact(self.train_path, artifact_path="datasets")

                model = self.model_training(X_train, y_train)
                self.save_model(model)

                metrics = self.evaluate_model(model, X_test, y_test)
                logger.info("Logging model parameters and metrics to mlflow")
                mlflow.log_params(model.named_steps["clf"].get_params())
                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(model, "spam_classification_pipeline")
                logger.info("Model pipeline completed and tracked on MLflow successfully!")
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Failed during model training pipeline", e) from e


if __name__ == "__main__":
    trainer = ModelTraining(PREPROCESS_TRAIN_DATA_PATH, PREPROCESS_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()