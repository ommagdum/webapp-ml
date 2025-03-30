from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
import ssl
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

model = joblib.load('spam_detector_model.joblib')

def preprocess_text(text):
    """
    Function to preprocess the text data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data or 'email_text' not in data:
                return jsonify({
                    "error": "No email text provided",
                    "status": "error"
                }), 400 

            email_text = data['email_text']
        
            processed_text = preprocess_text(email_text)
        
            prediction = model.predict([processed_text])[0]
            probability = model.predict_proba([processed_text])[0][1] 
        
            spam_label = "spam" if prediction == 1 else "ham"
            spam_probability = float(probability * 100)  # Convert numpy float to Python float

            return jsonify({
            "prediction": spam_label,
            "probability": spam_probability,
            "status": "success"
        })
    
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    app.run(debug=True)