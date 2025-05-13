import joblib
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_training_data(text_samples):
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
    X_text = [sample['text'] for sample in new_training_data]
    y_new = [sample['label'] for sample in new_training_data]
    
    X_new = preprocess_training_data(X_text)
    current_model.fit(X_new, y_new)
    
    predictions = current_model.predict(X_new)
    accuracy = accuracy_score(y_new, predictions)
    
    joblib.dump(current_model, save_path)
    
    return accuracy
