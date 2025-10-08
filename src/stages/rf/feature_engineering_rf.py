#!/usr/bin/env python3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import argparse
import sys
import os
import yaml

def preprocess_text(text, stop_words):
    """Clean the text: remove non-alphanumeric, lowercase, and remove stopwords."""
    if isinstance(text, str):
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        # Remove stopwords and join
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature engineering for Random Forest')
    parser.add_argument('--params', type=str, default='params.yaml', help='Path to parameters YAML file')
    
    args = parser.parse_args()
    
    try:
        # Load parameters
        with open(args.params, 'r') as f:
            params = yaml.safe_load(f)
        
        rf_params = params['rf']
        
        # Download stopwords if needed
        try:
            stop_words = set(stopwords.words(rf_params['featurize']['stop_words']))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words(rf_params['featurize']['stop_words']))
        
        # Load train and test data
        train_data = joblib.load('data/processed/rf/train_data.joblib')
        test_data = joblib.load('data/processed/rf/test_data.joblib')
        
        X_train = train_data['X_train']
        y_train = train_data['y_train']
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Preprocess text
        print("Preprocessing text for RF...")
        X_train_processed = X_train.apply(lambda x: preprocess_text(x, stop_words))
        X_test_processed = X_test.apply(lambda x: preprocess_text(x, stop_words))
        
        # Create TF-IDF features
        print("Creating TF-IDF features for RF...")
        vectorizer = TfidfVectorizer(
            max_features=rf_params['featurize']['max_features'],
            ngram_range=tuple(rf_params['featurize']['ngram_range'])
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train_processed)
        X_test_tfidf = vectorizer.transform(X_test_processed)
        
        # Save features and vectorizer
        os.makedirs('data/features/rf', exist_ok=True)
        
        joblib.dump(X_train_tfidf, 'data/features/rf/X_train_tfidf.joblib')
        joblib.dump(X_test_tfidf, 'data/features/rf/X_test_tfidf.joblib')
        joblib.dump(vectorizer, 'data/features/rf/vectorizer.joblib')
        
        print("RF Feature engineering completed!")
        print(f"Training features shape: {X_train_tfidf.shape}")
        print(f"Test features shape: {X_test_tfidf.shape}")
        
    except Exception as e:
        print(f"Error in RF feature engineering: {e}")
        sys.exit(1)