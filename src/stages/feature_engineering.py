import joblib
import yaml
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        if isinstance(text, str):
            text = re.sub(r'^[A-Z\s]+ \([A-Z\s]+\) - ', '', text)
            text = re.sub(r'^[A-Z\s]+\s\([A-Z\s]+\) – ', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()
            text = ' '.join([word for word in text.split() if word not in self.stop_words])
        return text

def engineer_features():
    """Perform feature engineering"""
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    max_features = params['feature_engineering']['max_features']
    ngram_range = tuple(params['feature_engineering']['ngram_range'])
    
    logger.info(f"Feature engineering with max_features={max_features}, ngram_range={ngram_range}")
    
    # Load data
    train_data = joblib.load('data/processed/train_data.joblib')
    test_data = joblib.load('data/processed/test_data.joblib')
    
    X_train = train_data['X_train']
    X_test = test_data['X_test']
    
    # Initialize preprocessor and vectorizer
    preprocessor = TextPreprocessor()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        preprocessor=preprocessor.preprocess_text
    )
    
    # Fit and transform
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save features and vectorizer
    joblib.dump(X_train_tfidf, 'data/processed/X_train_tfidf.joblib')
    joblib.dump(X_test_tfidf, 'data/processed/X_test_tfidf.joblib')
    joblib.dump(vectorizer, 'data/processed/vectorizer.joblib')
    
    logger.info(f"Feature engineering complete. Features: {X_train_tfidf.shape[1]}")

if __name__ == "__main__":
    engineer_features()