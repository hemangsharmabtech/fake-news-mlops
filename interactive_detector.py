import joblib
from src.stages.feature_engineering import TextPreprocessor

# Load model
model = joblib.load('model/lr_fake_news_model.joblib')
vectorizer = joblib.load('data/processed/vectorizer.joblib')
preprocessor = TextPreprocessor()

print("🎭 FAKE NEWS DETECTOR - INTERACTIVE MODE")
print("Type 'quit' to exit")
print("=" * 50)

while True:
    user_input = input("\n📝 Enter a news headline or article: ")
    
    if user_input.lower() == 'quit':
        break
        
    if user_input.strip():
        # Preprocess and predict
        processed_text = preprocessor.preprocess_text(user_input)
        features = vectorizer.transform([processed_text])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        if prediction == 0:
            print(f"�� FAKE NEWS ({(probability[0]*100):.1f}% confident)")
        else:
            print(f"✅ REAL NEWS ({(probability[1]*100):.1f}% confident)")
    else:
        print("Please enter some text!")
