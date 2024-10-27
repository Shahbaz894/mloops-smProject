import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging first
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Get environment variables with fallback to local MLflow
dagshub_username = os.getenv("MLFLOW_TRACKING_USERNAME")
dagshub_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub_token = os.getenv("DAGSHUB_PAT")
repo_owner = "Shahbaz894"
repo_name = "mloops-smProject"

# Check if we have DagHub credentials
use_dagshub = all([dagshub_username, dagshub_password, dagshub_token])

if use_dagshub:
    logger.info("Using DagHub MLflow tracking")
    os.environ['MLFLOW_TRACKING_URI'] = f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_password
else:
    logger.warning("DagHub credentials not found. Using local MLflow tracking.")
    os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000"

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise 

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    try:
        experiment_name = "dvc-pipeline"
        
        # Try to get or create the experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
                logger.debug(f'Created new experiment: {experiment_name}')
            else:
                logger.debug(f'Using existing experiment: {experiment_name}')
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            logger.warning("Continuing without MLflow tracking")
            experiment = None

        # Ensure directories exist
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./data/processed', exist_ok=True)
        os.makedirs('./reports', exist_ok=True)

        # Load and evaluate model
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_bow.csv')
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')

        # Only try MLflow logging if experiment setup was successful
        if experiment is not None:
            with mlflow.start_run():
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                    
                # Log parameters
                if hasattr(clf, 'get_params'):
                    params = clf.get_params()
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)

                # Log model and artifacts
                mlflow.sklearn.log_model(clf, 'model')
                save_model_info(mlflow.active_run().info.run_id, 'model', 'reports/experiment_info.json')
                mlflow.log_artifact('reports/metrics.json')
                mlflow.log_artifact('model_evaluation_errors.log')
        
        logger.debug('Model evaluation completed successfully')

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        raise

if __name__ == '__main__':
    main()