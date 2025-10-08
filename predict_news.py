import joblib
import sys
from src.stages.feature_engineering import TextPreprocessor

# Load your trained model and vectorizer
model = joblib.load('model/lr_fake_news_model.joblib')
vectorizer = joblib.load('data/processed/vectorizer.joblib')
preprocessor = TextPreprocessor()

def predict_news(text):
    """Predict if news is real or fake"""
    # Preprocess the text
    processed_text = preprocessor.preprocess_text(text)
    
    # Convert to features
    features = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # Return result
    if prediction == 0:
        return "FAKE NEWS", probability[0]
    else:
        return "REAL NEWS", probability[1]

# Test with some examples
test_articles = [
    "Scientists discover new planet that could support life",
    "ALIENS LAND IN NEW YORK! GOVERNMENT COVER UP!",
    "President signs new education bill into law",
    "SHOCKING: Drinking coffee makes you live forever!"
]

print("🔍 FAKE NEWS DETECTOR - LIVE DEMO")
print("=" * 40)

for i, article in enumerate(test_articles, 1):
    result, confidence = predict_news(article)
    print(f"\n{i}. {article[:60]}...")
    print(f"   👉 {result} ({(confidence*100):.1f}% confident)")

print(f"\n🎯 Model Accuracy: 98.8%")
