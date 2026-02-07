import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import yaml

import mlflow
import mlflow.sklearn


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_model(file_path: str):
    """Load the trained model from a file."""
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
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc)
        }
        
        logger.debug('Model evaluation metrics:')
        for metric, value in metrics_dict.items():
            logger.debug('  %s: %.4f', metric, value)
        
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def main():
    try:
        # Load parameters
        params = load_params(params_path='params.yaml')
        
        # Load model and test data
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_final.csv')
        
        # Split features and target
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate the model
        metrics = evaluate_model(clf, X_test, y_test)

        # ================== MLflow Tracking ==================
        # Set MLflow tracking to local file-based storage
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Employee-Attrition-Prediction")

        with mlflow.start_run(run_name="model_evaluation"):
            # Log parameters
            mlflow.log_param("model_type", clf.__class__.__name__)
            mlflow.log_param("n_estimators", params['model_building']['n_estimators'])
            mlflow.log_param("max_depth", params['model_building']['max_depth'])
            mlflow.log_param("random_state", params['model_building']['random_state'])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_test.shape[1])

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="model"
            )
            
            logger.debug('MLflow run completed')
        # =====================================================

        # Save metrics to reports
        save_metrics(metrics, 'reports/metrics.json')
        
        logger.info('Model evaluation completed successfully!')
        logger.info('Final Accuracy: %.4f', metrics['accuracy'])
        
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
