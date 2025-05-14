"""ML Model Training Module

This module provides functionality for text preprocessing and model training/retraining.
It includes utilities for cleaning and normalizing text data, as well as functions
for training machine learning models on new data.

The module implements NLP preprocessing techniques including:
- Text normalization (lowercase, whitespace normalization)
- Removal of emails and URLs
- Removal of non-alphabetic characters
- Tokenization
- Stopword removal
- Stemming

Typical usage example:
    new_data = [{"text": "Sample text", "label": 1}, ...]
    model = load_existing_model()
    accuracy = retrain_model(model, new_data, "path/to/save/model.pkl")
"""

import joblib
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_training_data(text_samples):
    """Preprocess text data for model training.
    
    Applies a series of text preprocessing steps to clean and normalize text data:
    1. Converts text to lowercase
    2. Removes email addresses
    3. Removes URLs
    4. Removes non-alphabetic characters
    5. Normalizes whitespace
    6. Tokenizes text
    7. Removes stopwords
    8. Applies stemming
    
    Args:
        text_samples (list): A list of text strings to preprocess.
        
    Returns:
        list: A list of preprocessed text strings ready for model training.
    """
    processed_samples = []
    for text in text_samples:
        text = text.lower()
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
        
        processed_samples.append(' '.join(tokens))
    return processed_samples

def retrain_model(current_model, new_training_data, save_path):
    """Retrain an existing model with new training data.
    
    Takes an existing scikit-learn compatible model and retrains it with new data.
    The function preprocesses the text data, fits the model, evaluates its accuracy,
    and saves the updated model to disk.
    
    Args:
        current_model: A scikit-learn compatible model object with a fit() method.
        new_training_data (list): A list of dictionaries, each containing 'text' and 'label' keys.
        save_path (str): File path where the retrained model will be saved.
        
    Returns:
        float: The accuracy score of the retrained model on the new training data.
        
    Note:
        This function evaluates accuracy on the same data used for training,
        which should only be used as a basic sanity check. For proper evaluation,
        use a separate validation set.
    """
    X_text = [sample['text'] for sample in new_training_data]
    y_new = [sample['label'] for sample in new_training_data]
    
    X_new = preprocess_training_data(X_text)
    current_model.fit(X_new, y_new)
    
    predictions = current_model.predict(X_new)
    accuracy = accuracy_score(y_new, predictions)
    
    joblib.dump(current_model, save_path)
    
    return accuracy
