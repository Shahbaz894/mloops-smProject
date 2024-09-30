import json
import mlflow
import logging
import os

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Shahbaz894"
repo_name = "mloops-smProject"  # Removed trailing space

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def get_latest_run_id(experiment_id: str) -> str:
    try:
        client = mlflow.MlflowClient()
        runs = client.list_run_infos(experiment_id)
        if not runs:
            raise ValueError(f"No runs found for the specified experiment: {experiment_id}")
        latest_run = max(runs, key=lambda run: run.start_time)
        logger.debug(f"Latest run ID for experiment {experiment_id}: {latest_run.run_id}")
        return latest_run.run_id
    except Exception as e:
        logger.error(f"Error fetching the latest run ID: {e}")
        raise

def register_model(model_name: str, model_info: dict):
    try:
        # Get the latest run ID if not available in model_info
        run_id = model_info.get('run_id') or get_latest_run_id(model_info['experiment_id'])
        model_uri = f"runs:/{run_id}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        # Use MlflowClient to transition the model to the staging area
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
