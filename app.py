from ml.management.model_manager import ModelManager
import threading
import time
import os
import re
import joblib
import nltk
import ssl
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask_cors import CORS
from routes.ml_routes import ml_routes
from ml.shared import model_manager


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

# Global variables for model management
model = None
current_model_version = None
model_manager = ModelManager(models_dir='models')

def initialize_model_manager():
    """Initialize the model manager with the initial model if needed"""
    try:
        # Check if metadata exists
        if not os.path.exists(model_manager.metadata_file):
            # Check if initial_model.pkl exists in models directory
            initial_model_path = os.path.join(model_manager.models_dir, 'initial_model.pkl')
            if os.path.exists(initial_model_path):
                # Initialize metadata with this model
                model_manager.add_version('initial_model', 0.9)  # Assuming default accuracy
                print("Initialized metadata with initial_model")
            else:
                # Check if we have a default model
                default_model_path = 'spam_detector_model.joblib'
                if os.path.exists(default_model_path):
                    # Copy default model to models directory
                    os.makedirs(model_manager.models_dir, exist_ok=True)
                    new_model_path = os.path.join(model_manager.models_dir, 'initial_model.pkl')
                    joblib.dump(joblib.load(default_model_path), new_model_path)
                    model_manager.add_version('initial_model', 0.9)
                    print(f"Copied default model to {new_model_path} and initialized metadata")
    except Exception as e:
        print(f"Error initializing model manager: {str(e)}")

def load_model():
    """Load the current model using the ModelManager"""
    global model, current_model_version
    
    try:
        metadata = model_manager.load_metadata()
        version = metadata.get('current_version')
        
        # Only reload if version changed
        if version != current_model_version:
            print(f"Loading model version: {version}")
            model_path = os.path.join(model_manager.models_dir, f"{version}.pkl")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                current_model_version = version
                print(f"Successfully loaded model version: {version}")
            else:
                print(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # If this is initial load and failed, try to load the default model
        if current_model_version is None:
            try:
                # Fallback to the original model path
                print("Attempting to load default model...")
                default_model_path = 'spam_detector_model.joblib'
                if os.path.exists(default_model_path):
                    model = joblib.load(default_model_path)
                    current_model_version = "default"
                    print("Successfully loaded default model")
                else:
                    print(f"Default model not found at: {default_model_path}")
                    raise FileNotFoundError(f"Default model not found at: {default_model_path}")
            except Exception as fallback_error:
                print(f"Error loading default model: {str(fallback_error)}")
                raise
    
    return current_model_version

def model_watcher():
    """Background thread that periodically checks for model updates"""
    while True:
        try:
            load_model()
        except Exception as e:
            print(f"Error in model watcher: {str(e)}")
        
        # Check every 60 seconds
        time.sleep(60)

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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Register blueprints
app.register_blueprint(ml_routes, url_prefix='/ml')

# Initialize model manager and model on startup
initialize_model_manager()
try:
    load_model()
    if model is None:
        print("WARNING: Model failed to load on startup!")
except Exception as e:
    print(f"ERROR during model initialization: {str(e)}")

# Start the background thread for model watching
model_thread = threading.Thread(target=model_watcher, daemon=True)
model_thread.start()

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'up',
        'message': 'ML service is running',
        'model_version': current_model_version,
        'model_status': model_status
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            # Try to load the model again
            load_model()
            if model is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded. Please try again later.'
                }), 503
        
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
                'probability': float(probability),
                'model_version': current_model_version
            }
        })
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# If running directly (not imported)
if __name__ == '__main__':
    print("Starting ML service on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)
