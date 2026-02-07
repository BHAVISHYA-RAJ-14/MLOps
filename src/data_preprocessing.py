import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Ordinal encoding mappings for ordered categorical variables
ORDINAL_MAPPINGS = {
    'Work-Life Balance': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3},
    'Job Satisfaction': {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3},
    'Performance Rating': {'Below Average': 0, 'Average': 1, 'High': 2},
    'Job Level': {'Entry': 0, 'Mid': 1, 'Senior': 2},
    'Company Size': {'Small': 0, 'Medium': 1, 'Large': 2},
    'Company Reputation': {'Poor': 0, 'Fair': 1, 'Good': 2},
    'Employee Recognition': {'Low': 0, 'Medium': 1, 'High': 2}
}

# Binary encoding mappings
BINARY_MAPPINGS = {
    'Gender': {'Male': 0, 'Female': 1},
    'Overtime': {'No': 0, 'Yes': 1},
    'Remote Work': {'No': 0, 'Yes': 1},
    'Leadership Opportunities': {'No': 0, 'Yes': 1},
    'Innovation Opportunities': {'No': 0, 'Yes': 1},
    'Attrition': {'Stayed': 0, 'Left': 1}
}

# Columns for one-hot encoding
ONEHOT_COLUMNS = ['Job Role', 'Education Level', 'Marital Status']


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal categorical features with proper ordering."""
    try:
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                logger.debug('Ordinal encoded column: %s', col)
        return df
    except Exception as e:
        logger.error('Error encoding ordinal features: %s', e)
        raise


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary categorical features."""
    try:
        for col, mapping in BINARY_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                logger.debug('Binary encoded column: %s', col)
        return df
    except Exception as e:
        logger.error('Error encoding binary features: %s', e)
        raise


def encode_onehot_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categorical features."""
    try:
        for col in ONEHOT_COLUMNS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col.replace(' ', '_'), drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                logger.debug('One-hot encoded column: %s', col)
        return df
    except Exception as e:
        logger.error('Error one-hot encoding features: %s', e)
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    try:
        original_len = len(df)
        df = df.drop_duplicates(keep='first')
        removed = original_len - len(df)
        if removed > 0:
            logger.debug('Removed %d duplicate rows', removed)
        return df
    except Exception as e:
        logger.error('Error removing duplicates: %s', e)
        raise


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to a dataframe."""
    try:
        logger.debug('Starting preprocessing for DataFrame with shape %s', df.shape)
        
        # Remove duplicates
        df = remove_duplicates(df)
        
        # Encode ordinal features
        df = encode_ordinal_features(df)
        
        # Encode binary features
        df = encode_binary_features(df)
        
        # One-hot encode nominal features
        df = encode_onehot_features(df)
        
        logger.debug('Preprocessing completed. Final shape: %s', df.shape)
        return df
    except Exception as e:
        logger.error('Error during preprocessing: %s', e)
        raise


def main():
    """Main function to load raw data, preprocess it, and save the processed data."""
    try:
        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded: train=%s, test=%s', train_data.shape, test_data.shape)
        
        # Preprocess both datasets
        train_processed = preprocess_df(train_data.copy())
        test_processed = preprocess_df(test_data.copy())
        
        # Ensure both datasets have the same columns (align columns)
        train_cols = set(train_processed.columns)
        test_cols = set(test_processed.columns)
        
        # Add missing columns with 0 values
        for col in train_cols - test_cols:
            test_processed[col] = 0
        for col in test_cols - train_cols:
            train_processed[col] = 0
        
        # Reorder columns to match
        all_cols = sorted(train_processed.columns.tolist())
        train_processed = train_processed[all_cols]
        test_processed = test_processed[all_cols]
        
        logger.debug('Columns aligned. Train: %s, Test: %s', train_processed.shape, test_processed.shape)
        
        # Save processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.info('Data preprocessing completed successfully!')
        logger.debug('Processed data saved to %s', data_path)
        
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
