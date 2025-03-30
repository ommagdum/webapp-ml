from flask import Flask, request, jsonify
import joblib
import re
import nltk
import ssl
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the model
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
    try:
        data = request.get_json()
        
        if not data or 'email_text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing email_text'
            }), 400
        
        email_text = data['email_text']
        
        if not email_text:
            return jsonify({
                'success': False,
                'error': 'Empty email_text'
            }), 400
        
        processed_text = preprocess_text(email_text)
        
        prediction = model.predict([processed_text])[0]
        probability = model.predict_proba([processed_text])[0][1]
        
        return jsonify({
            'success': True,
            'data': {
                'prediction': int(prediction),
                'probability': float(probability)
            }
        })
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'up',
        'message': 'ML service is running'
    })

if __name__ == '__main__':
    print("Starting ML service on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)
