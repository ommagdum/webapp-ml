"""ML Service API Application

This module implements a Flask-based REST API service for machine learning predictions.
It provides endpoints for spam detection, model health checks, and integrates with the
model management system for automatic model updates and versioning.

The application includes:
- Dynamic model loading with version tracking
- Background thread for model updates monitoring
- Text preprocessing pipeline for prediction requests
- RESTful API endpoints for predictions and health checks
- Error handling and graceful degradation

The service automatically initializes with the latest model version and
periodically checks for model updates without requiring a restart.
"""

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
# Initialize model manager
model_manager = ModelManager(models_dir='models')

def get_model():
    """Thread-safe model getter that ensures proper initialization"""
    if not hasattr(get_model, "model"):
        get_model.model = None
        get_model.version = None
    
    if get_model.model is None:
        try:
            metadata = model_manager.load_metadata()
            version = metadata.get('current_version')
            
            if version is None:
                print("No model version found in metadata. Using initial_model.")
                version = 'initial_model'
                model_path = os.path.join(model_manager.models_dir, f"{version}.pkl")
            elif version != get_model.version:
                print(f"Loading model version: {version}")
                model_path = os.path.join(model_manager.models_dir, f"{version}.pkl")
                
                if os.path.exists(model_path):
                    get_model.model = joblib.load(model_path)
                    
                    # Check if model is a pipeline with a vectorizer
                    if hasattr(get_model.model, 'named_steps') and 'vect' in get_model.model.named_steps:
                        if not hasattr(get_model.model.named_steps['vect'], 'idf_') or get_model.model.named_steps['vect'].idf_ is None:
                            print("Warning: TF-IDF vectorizer not fitted. Attempting to initialize...")
                            get_model.model.named_steps['vect'].fit(["dummy text for initialization"])
                    elif hasattr(get_model.model, 'vectorizer') and hasattr(get_model.model.vectorizer, 'idf_'):
                        if get_model.model.vectorizer.idf_ is None:
                            print("Warning: TF-IDF vectorizer not fitted. Attempting to initialize...")
                            get_model.model.vectorizer.fit(["dummy text for initialization"])
                    
                    get_model.version = version
                    print(f"Successfully loaded model version: {version}")
                else:
                    print(f"Model file not found: {model_path}")
                    raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    return get_model.model, version

def initialize_model_manager():
    """Initialize the model manager with the initial model if needed.
    
    This function checks if the model metadata exists and initializes it if needed.
    It follows this process:
    1. Check if metadata file exists
    2. If not, look for an initial_model.pkl in the models directory
    3. If that's not found, look for a default model (spam_detector_model.joblib)
    4. If found, copy it to the models directory and initialize metadata
    
    The function handles the bootstrapping process for the model management system,
    ensuring that a valid model is available when the service starts.
    """
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
    """Load the current model using the ModelManager.
    
    This function loads the current model version from the model manager.
    It only reloads the model if the version has changed since the last load.
    If loading fails and no model is currently loaded, it attempts to fall back
    to a default model to ensure the service remains operational.
    
    The function handles:
    - Checking the current model version in metadata
    - Loading the model only if version has changed (optimization)
    - Graceful fallback to default model if primary loading fails
    - Comprehensive error handling and logging
    
    Returns:
        str: The version ID of the currently loaded model
        
    Raises:
        FileNotFoundError: If no model can be found (primary or fallback)
        Exception: For other errors during model loading that cannot be recovered from
    """
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
    """Background thread that periodically checks for model updates.
    
    Runs as a daemon thread to periodically check for model updates
    by calling load_model() every 60 seconds. This ensures the service
    always uses the most recent model version without requiring a restart.
    
    If an error occurs during model loading, it is logged but the thread
    continues running to try again in the next cycle.
    """
    while True:
        try:
            load_model()
        except Exception as e:
            print(f"Error in model watcher: {str(e)}")
        
        # Check every 60 seconds
        time.sleep(60)

def preprocess_text(text):
    """
    Preprocess text data for model prediction.
    
    Applies a series of text preprocessing steps to clean and normalize text data
    before passing it to the prediction model. The preprocessing pipeline includes:
    
    1. Lowercase conversion - Normalizes case differences
    2. Email removal - Removes email addresses which could contain personal information
    3. URL removal - Removes web links which are not relevant for classification
    4. Special character removal - Keeps only alphabetic characters and spaces
    5. Whitespace normalization - Standardizes spacing between words
    6. Tokenization - Splits text into individual words
    7. Stopword removal - Removes common words that don't add meaning
    8. Stemming - Reduces words to their root form
    
    Args:
        text (str): The raw text to preprocess
        
    Returns:
        str: The preprocessed text ready for model prediction
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
    """Health check endpoint for the ML service.
    
    Returns information about the service status, including whether
    the model is loaded and which version is currently active.
    
    Returns:
        JSON: A response containing service status information:
            - status: 'up' if service is running
            - message: Description of service status
            - model_version: Current active model version
            - model_status: Whether model is loaded or not
    """
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'up',
        'message': 'ML service is running',
        'model_version': current_model_version,
        'model_status': model_status
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for spam detection.
    
    Accepts a POST request with JSON payload containing 'email_text',
    processes the text, and returns a prediction (spam or not spam)
    along with the probability score.
    
    Request Body:
        JSON with the following fields:
            - email_text: The text content to analyze for spam detection
    
    Returns:
        JSON: A response containing:
            - success: Boolean indicating if the request was successful
            - data: (if successful)
                - prediction: 1 for spam, 0 for not spam
                - probability: Confidence score (0-1) for spam classification
                - model_version: Version of the model used for prediction
            - error: (if unsuccessful) Error message
            
    Response Codes:
        - 200: Successful prediction
        - 400: Missing or invalid input
        - 500: Server error during processing
        - 503: Model not available
    """
    try:
        # Get model instance (thread-safe)
        model, version = get_model()
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not available'
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
                'model_version': get_model.version
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
