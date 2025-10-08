#!/usr/bin/env python3
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import argparse
import sys
import os
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Random Forest model')
    parser.add_argument('--params', type=str, default='params.yaml', help='Path to parameters YAML file')
    
    args = parser.parse_args()
    
    try:
        # Load parameters
        with open(args.params, 'r') as f:
            params = yaml.safe_load(f)
        
        rf_params = params['rf']
        
        # Load model and test data
        rf_model = joblib.load('models/rf/rf_fake_news_model.joblib')
        X_test_tfidf = joblib.load('data/features/rf/X_test_tfidf.joblib')
        test_data = joblib.load('data/processed/rf/test_data.joblib')
        y_test = test_data['y_test']
        
        # Make predictions
        print("Making predictions with RF model...")
        y_pred = rf_model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, 
                                          target_names=rf_params['evaluate']['target_names'],
                                          output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            "model_type": "random_forest",
            "accuracy": round(accuracy, 4),
            "precision_fake": round(report_dict['Fake']['precision'], 4),
            "recall_fake": round(report_dict['Fake']['recall'], 4),
            "f1_fake": round(report_dict['Fake']['f1-score'], 4),
            "precision_true": round(report_dict['True']['precision'], 4),
            "recall_true": round(report_dict['True']['recall'], 4),
            "f1_true": round(report_dict['True']['f1-score'], 4),
            "confusion_matrix": cm.tolist(),
            "training_samples": int(len(y_test) / rf_params['data']['test_size'] * (1 - rf_params['data']['test_size'])),
            "test_samples": len(y_test),
            "n_estimators": rf_params['train']['n_estimators']
        }
        
        # Save metrics
        os.makedirs('metrics/rf', exist_ok=True)
        with open('metrics/rf/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("RF Model evaluation completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Metrics saved to: metrics/rf/metrics.json")
        
        # Print detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=rf_params['evaluate']['target_names']))
        
    except Exception as e:
        print(f"Error in RF model evaluation: {e}")
        sys.exit(1)