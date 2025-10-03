import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare the dataset"""
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    sample_size = params['data_ingestion']['sample_size_per_class']
    test_size = params['data_ingestion']['test_size']
    random_state = params['data_ingestion']['random_state']
    
    logger.info(f"Loading data with sample_size={sample_size}, test_size={test_size}")
    
    # Load datasets
    df_fake = pd.read_csv('data/raw/Fake.csv').head(sample_size)
    df_true = pd.read_csv('data/raw/True.csv').head(sample_size)
    
    # Add labels and combine
    df_fake['label'] = 0
    df_true['label'] = 1
    df = pd.concat([df_fake, df_true], ignore_index=True)
    
    # Combine title and text
    df['content'] = df['title'] + ' ' + df['text']
    
    # Split features and target
    X = df['content']
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save processed data
    joblib.dump({'X_train': X_train, 'y_train': y_train}, 'data/processed/train_data.joblib')
    joblib.dump({'X_test': X_test, 'y_test': y_test}, 'data/processed/test_data.joblib')
    
    logger.info(f"Data ingestion complete. Train: {len(X_train)}, Test: {len(X_test)}")

if __name__ == "__main__":
    load_data()