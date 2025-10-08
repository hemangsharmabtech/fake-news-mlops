#!/bin/bash

echo "🎓 FAKE NEWS DETECTOR - LIVE DEMO"
echo "=================================="

echo -e "\n1. 📊 CURRENT MODEL STATUS:"
cat model/metrics.json

echo -e "\n2. 🔍 TESTING THE MODEL:"
python -c "
import joblib
from src.stages.feature_engineering import TextPreprocessor

model = joblib.load('model/lr_fake_news_model.joblib')
vectorizer = joblib.load('data/processed/vectorizer.joblib')
preprocessor = TextPreprocessor()

test_news = [
    'Breaking: Scientists make major discovery in cancer research',
    'SHOCKING: The moon is made of cheese! NASA confirms!',
    'New education policy announced by government officials',
    'CELEBRITY SELLS SOUL TO DEVIL FOR FAME AND FORTUNE!'
]

for news in test_news:
    processed = preprocessor.preprocess_text(news)
    features = vectorizer.transform([processed])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    result = 'FAKE' if pred == 0 else 'REAL'
    confidence = proba[0] if pred == 0 else proba[1]
    print(f'📰 {news[:50]}... → {result} ({(confidence*100):.1f}%)')
"

echo -e "\n3. 🔄 REPRODUCIBILITY DEMO:"
echo "   To re-run everything: dvc repro"
echo "   To try different settings: dvc exp run -S param=value"

echo -e "\n4. 📈 EXPERIMENT RESULTS:"
dvc exp show --no-pager | head -8
