cat > README.md << 'EOF'
# Fake News Detection - MLOps Pipeline

A complete MLOps pipeline for fake news detection using DVC (Data Version Control) and S3 storage.

## 🚀 Features

- **Two Models**: Linear Regression (98.8% accuracy) and Random Forest (99.6% accuracy)
- **DVC Pipeline**: 7-stage reproducible workflow
- **S3 Integration**: Cloud storage for large artifacts
- **Experiment Tracking**: Metrics comparison and versioning
- **Parameter Management**: Centralized configuration with `params.yaml`

## 📊 Results

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) |
|-------|----------|------------------|---------------|-----------------|
| Linear Regression | 98.8% | - | - | - |
| Random Forest | 99.6% | 99.75% | 99.45% | 99.6% |

## 🛠️ Installation

```bash
# Clone repository
git clone <your-repo-url>
cd fake_news_mlops

# Install dependencies
pip install -r requirements.txt

# Pull data and models
dvc pull -r myremote# CI/CD Test
 
Testing GitHub Secrets setup
