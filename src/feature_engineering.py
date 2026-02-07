import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import logging
import pickle


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Numerical columns to scale
NUMERICAL_COLUMNS = [
    'Age', 'Years at Company', 'Monthly Income', 'Number of Promotions',
    'Distance from Home', 'Number of Dependents', 'Company Tenure'
]


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features to improve model accuracy."""
    try:
        # Income per year at company
        df['Income_Per_Year'] = df['Monthly Income'] / np.maximum(df['Years at Company'], 1)
        logger.debug('Created Income_Per_Year feature')
        
        # Tenure ratio - how long in current company vs total tenure
        df['Tenure_Ratio'] = df['Years at Company'] / np.maximum(df['Company Tenure'], 1)
        # Cap at 1 (can't work longer at company than total tenure)
        df['Tenure_Ratio'] = df['Tenure_Ratio'].clip(0, 1)
        logger.debug('Created Tenure_Ratio feature')
        
        # Age groups: Young (18-30), Mid (31-45), Senior (46+)
        df['Age_Group_Mid'] = ((df['Age'] >= 31) & (df['Age'] <= 45)).astype(int)
        df['Age_Group_Senior'] = (df['Age'] > 45).astype(int)
        logger.debug('Created Age_Group features')
        
        # Promotions per year
        df['Promotions_Per_Year'] = df['Number of Promotions'] / np.maximum(df['Years at Company'], 1)
        logger.debug('Created Promotions_Per_Year feature')
        
        return df
    except Exception as e:
        logger.error('Error creating derived features: %s', e)
        raise


def scale_numerical_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Scale numerical features using StandardScaler."""
    try:
        scaler = StandardScaler()
        
        # Get columns that exist in the dataframe
        cols_to_scale = [col for col in NUMERICAL_COLUMNS if col in train_df.columns]
        
        # Add derived numerical columns
        derived_cols = ['Income_Per_Year', 'Tenure_Ratio', 'Promotions_Per_Year']
        cols_to_scale.extend([col for col in derived_cols if col in train_df.columns])
        
        if cols_to_scale:
            # Fit on training data only
            train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
            test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
            logger.debug('Scaled columns: %s', cols_to_scale)
            
            # Save the scaler for future use
            os.makedirs('models', exist_ok=True)
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            logger.debug('Scaler saved to models/scaler.pkl')
        
        return train_df, test_df
    except Exception as e:
        logger.error('Error scaling features: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        # Load preprocessed data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        # Create derived features
        train_data = create_derived_features(train_data)
        test_data = create_derived_features(test_data)
        
        # Scale numerical features
        train_data, test_data = scale_numerical_features(train_data, test_data)
        
        # Ensure Attrition column is last (as label)
        if 'Attrition' in train_data.columns:
            cols = [c for c in train_data.columns if c != 'Attrition'] + ['Attrition']
            train_data = train_data[cols]
            test_data = test_data[cols]
        
        # Save processed data
        save_data(train_data, './data/processed/train_final.csv')
        save_data(test_data, './data/processed/test_final.csv')
        
        logger.info('Feature engineering completed successfully!')
        logger.debug('Final train shape: %s, test shape: %s', train_data.shape, test_data.shape)
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
