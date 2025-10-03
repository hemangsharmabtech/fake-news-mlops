import joblib
import yaml
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """Train and evaluate the model"""
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    model_name = params['model_building']['model_name']
    solver = params['model_building']['solver']
    max_iter = params['model_building']['max_iter']
    
    logger.info(f"Training model: {model_name} with solver={solver}, max_iter={max_iter}")
    
    # Load features
    X_train_tfidf = joblib.load('data/processed/X_train_tfidf.joblib')
    X_test_tfidf = joblib.load('data/processed/X_test_tfidf.joblib')
    train_data = joblib.load('data/processed/train_data.joblib')
    test_data = joblib.load('data/processed/test_data.joblib')
    
    y_train = train_data['y_train']
    y_test = test_data['y_test']
    
    # Train model
    if model_name == "logistic_regression":
        model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "model": model_name,
        "parameters": {
            "solver": solver,
            "max_iter": max_iter
        }
    }
    
    with open('model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    joblib.dump(model, 'model/lr_fake_news_model.joblib')
    
    logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()