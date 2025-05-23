# 🧠 ML Model & Training

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=nltk&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

## Model Architecture

The spam detection system uses a machine learning pipeline with the following components:

- **Text Preprocessing**: Tokenization, stopword removal, and stemming using NLTK
- **Feature Extraction**: TF-IDF vectorization
- **Classification**: Linear SVM classifier with probability calibration
- **Model Versioning**: Automatic version control with rollback capabilities

## 🛠️ Preprocessing Pipeline

```python
# Example preprocessing steps
1. Text cleaning (lowercase, remove special chars)
2. Tokenization
3. Stopword removal
4. Stemming/Lemmatization
5. TF-IDF vectorization
```

## 🏋️ Training & Retraining

The model supports automated retraining with new data while maintaining version control:

- **Version Management**: Track model versions with metadata
- **Auto-rollback**: Reverts to previous version if accuracy drops
- **Background Training**: Non-blocking model updates
- **Metrics Tracking**: Logs accuracy, precision, recall, and F1-score

## 🚀 Deployment

The model is deployed as a Flask service with the following endpoints:

- `POST /predict`: Get spam/ham predictions
- `POST /retrain`: Trigger model retraining with new data
- `GET /version`: Check current model version and metrics

## 📦 Dependencies

```plaintext
scikit-learn>=1.5.1
nltk>=3.6.5
numpy>=1.23.5
joblib>=1.1.1
flask>=2.0.1
```

## 🔄 Model Improvement

To update or improve the model:

### 1. Data Collection
- Add more labeled examples to `training_data/`
- Ensure class balance between spam/ham

### 2. Feature Engineering
- Add new text features
- Experiment with different n-gram ranges
- Try advanced word embeddings

### 3. Model Tuning
- Update hyperparameters in `train_model.py`
- Test different classifiers
- Adjust class weights for imbalanced data

### 4. Deploy Updates
- New models are automatically versioned
- System validates performance before activation
- Fallback to previous version if needed

## 📊 Performance Monitoring

- Model metrics are logged to `logs/model_metrics.log`
- Performance dashboards can be integrated via the `/metrics` endpoint
- Automatic alerts for performance degradation
