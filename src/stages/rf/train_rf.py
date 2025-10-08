#!/usr/bin/env python3
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
import sys
import os
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Random Forest model')
    parser.add_argument('--params', type=str, default='params.yaml', help='Path to parameters YAML file')
    
    args = parser.parse_args()
    
    try:
        # Load parameters
        with open(args.params, 'r') as f:
            params = yaml.safe_load(f)
        
        rf_params = params['rf']
        
        # Load training features
        X_train_tfidf = joblib.load('data/features/rf/X_train_tfidf.joblib')
        train_data = joblib.load('data/processed/rf/train_data.joblib')
        y_train = train_data['y_train']
        
        # Train Random Forest model
        print("Training Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=rf_params['train']['n_estimators'],
            max_depth=rf_params['train']['max_depth'],
            min_samples_split=rf_params['train']['min_samples_split'],
            min_samples_leaf=rf_params['train']['min_samples_leaf'],
            random_state=rf_params['train']['random_state'],
            n_jobs=rf_params['train']['n_jobs']
        )
        
        rf_model.fit(X_train_tfidf, y_train)
        
        # Save model
        os.makedirs('models/rf', exist_ok=True)
        joblib.dump(rf_model, 'models/rf/rf_fake_news_model.joblib')
        
        print("RF Model training completed!")
        print(f"Model saved to: models/rf/rf_fake_news_model.joblib")
        
    except Exception as e:
        print(f"Error in RF model training: {e}")
        sys.exit(1)