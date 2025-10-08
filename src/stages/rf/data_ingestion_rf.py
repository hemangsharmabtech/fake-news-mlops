#!/usr/bin/env python3
import pandas as pd
import joblib
import argparse
import sys
import os
import yaml
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data ingestion for Random Forest')
    parser.add_argument('--params', type=str, default='params.yaml', help='Path to parameters YAML file')
    
    args = parser.parse_args()
    
    try:
        # Load parameters
        with open(args.params, 'r') as f:
            params = yaml.safe_load(f)
        
        rf_params = params['rf']
        
        print("Loading data for Random Forest experiment...")
        
        # Load data with sampling
        sample_size = rf_params['data']['sample_size_per_class']
        df_fake = pd.read_csv('data/raw/Fake.csv').head(sample_size)
        df_true = pd.read_csv('data/raw/True.csv').head(sample_size)
        
        # Add labels
        df_fake['label'] = 0
        df_true['label'] = 1
        
        # Combine and shuffle
        df = pd.concat([df_fake, df_true], ignore_index=True)
        df = df.sample(frac=1, random_state=rf_params['data']['random_state']).reset_index(drop=True)
        
        # Drop metadata to prevent leakage
        df = df.drop(columns=['subject', 'date'], errors='ignore')
        
        # Combine title and text
        df['content'] = df['title'] + ' ' + df['text']
        
        # Split data
        X = df['content']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=rf_params['data']['test_size'], 
            random_state=rf_params['data']['random_state'], 
            stratify=y
        )
        
        # Save train and test data
        os.makedirs('data/processed/rf', exist_ok=True)
        
        train_data = {
            'X_train': X_train,
            'y_train': y_train
        }
        
        test_data = {
            'X_test': X_test,
            'y_test': y_test
        }
        
        joblib.dump(train_data, 'data/processed/rf/train_data.joblib')
        joblib.dump(test_data, 'data/processed/rf/test_data.joblib')
        
        print("RF Data ingestion completed!")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
    except Exception as e:
        print(f"Error in RF data ingestion: {e}")
        sys.exit(1)