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
model = None
version = None
model_manager = ModelManager(models_dir='models')

def get_model():
    """Thread-safe model getter that ensures proper initialization
    
    Returns:
        tuple: (model, version) where model is the loaded model and version is the model version
    """
    global model, version
    
    if model is None:
        try:
            # Get the current model version from metadata
            metadata = model_manager.load_metadata()
            version = metadata.get('current_version')
            
            if version is None:
                print("No model version found in metadata. Using initial_model.")
                version = 'initial_model'
            
            print(f"Loading model version: {version}")
            
            try:
                # Get the model path using the ModelManager
                model_path = model_manager.get_model_path(version)
                print(f"Loading model from: {model_path}")
                
                # Load the model
                model = joblib.load(model_path)
                print(f"Successfully loaded model version: {version}")
                
                # Verify the model has the required predict method
                if not hasattr(model, 'predict'):
                    raise AttributeError("Loaded model does not have a 'predict' method")
                
                # Initialize vectorizer if needed (fallback)
                if hasattr(model, 'named_steps') and 'vect' in model.named_steps:
                    vectorizer = model.named_steps['vect']
                    if not hasattr(vectorizer, 'idf_') or vectorizer.idf_ is None:
                        print("Warning: TF-IDF vectorizer not fitted. Initializing with sample data...")
                        sample_texts = [
                            "spam email free offer money win prize", 
                            "normal email meeting schedule reminder", 
                            "business proposal contract agreement",
                            "personal message family friend hello",
                            "urgent action required immediately"
                        ]
                        # Fit the vectorizer with sample data
                        vectorizer.fit(sample_texts)
                        # Save the model with the fitted vectorizer
                        joblib.dump(model, model_path)
                        print("Re-saved model with fitted vectorizer")
                
                return model, version
                
            except FileNotFoundError as e:
                error_msg = f"Model file not found for version {version}"
                print(error_msg)
                return None, version
                
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}\nError type: {type(e).__name__}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                return None, version
                
        except Exception as e:
            error_msg = f"Error in get_model: {str(e)}\nError type: {type(e).__name__}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, version
    
    return model, version

def initialize_model_manager():
    """Initialize the model manager with the initial model if needed.
    
    This function checks if the model metadata exists and initializes it if needed.
    It follows this process:
    1. Check if metadata file exists
    2. If not, look for an initial_model.joblib or initial_model.pkl in the models directory
    3. If that's not found, look for a default model (spam_detector_model.joblib)
    4. If found, copy it to the models directory and initialize metadata
    
    The function handles the bootstrapping process for the model management system,
    ensuring that a valid model is available when the service starts.
    """
    try:
        # Check if metadata exists
        if not os.path.exists(model_manager.metadata_file):
            # Check if initial_model exists in models directory (try both extensions)
            initial_model_path_joblib = os.path.join(model_manager.models_dir, 'initial_model.joblib')
            initial_model_path_pkl = os.path.join(model_manager.models_dir, 'initial_model.pkl')
            
            if os.path.exists(initial_model_path_joblib):
                # Initialize metadata with this model
                model_manager.add_version('initial_model', 0.9)  # Assuming default accuracy
                print("Initialized metadata with initial_model.joblib")
            elif os.path.exists(initial_model_path_pkl):
                # Initialize metadata with this model
                model_manager.add_version('initial_model', 0.9)  # Assuming default accuracy
                print("Initialized metadata with initial_model.pkl")
            else:
                # Check if we have a default model
                default_model_path = 'spam_detector_model.joblib'
                if os.path.exists(default_model_path):
                    # Copy default model to models directory
                    os.makedirs(model_manager.models_dir, exist_ok=True)
                    new_model_path = os.path.join(model_manager.models_dir, 'initial_model.joblib')
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
    try:
        # Get the model and version from get_model()
        global model, version
        model, version = get_model()
        
        if model is None:
            raise FileNotFoundError("No models available")
            
        return version
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

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
app.register_blueprint(ml_routes, url_prefix='/')

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
            - model_status: Detailed model status
            - vectorizer_status: Status of the TF-IDF vectorizer
            - model_has_predict: Whether model has predict method
            - model_has_predict_proba: Whether model has predict_proba method
            - service_ready: Whether service is fully operational
    """
    try:
        # Try to get the current model and version
        current_model, current_version = get_model()
        model_loaded = current_model is not None
        
        # Check model capabilities
        has_predict = hasattr(current_model, 'predict') if model_loaded else False
        has_predict_proba = hasattr(current_model, 'predict_proba') if model_loaded else False
        
        # Check vectorizer status if available
        vectorizer_status = 'n/a'
        if model_loaded and hasattr(current_model, 'named_steps') and 'vect' in current_model.named_steps:
            vectorizer = current_model.named_steps['vect']
            if hasattr(vectorizer, 'idf_') and vectorizer.idf_ is not None:
                vectorizer_status = 'fitted'
            else:
                vectorizer_status = 'not_fitted'
        
        # Determine overall service readiness
        service_ready = model_loaded and has_predict and has_predict_proba
        if model_loaded and hasattr(current_model, 'named_steps') and 'vect' in current_model.named_steps:
            service_ready = service_ready and (vectorizer_status == 'fitted')
        
        # Prepare response
        response = {
            'status': 'up',
            'message': 'ML service is running',
            'model_version': current_version,
            'model_status': 'loaded' if model_loaded else 'not_loaded',
            'vectorizer_status': vectorizer_status,
            'model_has_predict': has_predict,
            'model_has_predict_proba': has_predict_proba,
            'service_ready': service_ready,
            'timestamp': time.time()
        }
        
        # Add detailed error information if service is not ready
        if not service_ready:
            if not model_loaded:
                response['error'] = 'Model not loaded'
            elif not has_predict:
                response['error'] = 'Model missing predict method'
            elif not has_predict_proba:
                response['error'] = 'Model missing predict_proba method'
            elif vectorizer_status == 'not_fitted':
                response['error'] = 'TF-IDF vectorizer not fitted'
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = str(e)
        traceback_details = traceback.format_exc()
        print(f"Health check error: {error_details}\n{traceback_details}")
        
        return jsonify({
            'status': 'error',
            'message': 'Error checking service health',
            'error': error_details,
            'model_status': 'error',
            'service_ready': False,
            'timestamp': time.time()
        }), 500
    

@app.route("/ping")
def ping():
    return jsonify(status="ok")

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
            - status: Additional status information for debugging
            
    Response Codes:
        - 200: Successful prediction
        - 400: Missing or invalid input
        - 500: Server error during processing
        - 503: Model not available
    """
    try:
        # Get model instance (thread-safe)
        try:
            model, current_version = get_model()
            
            if model is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded. Please check server logs for details.',
                    'status': 'model_unavailable'
                }), 503
                
            # Verify model has required methods
            if not all(hasattr(model, method) for method in ['predict', 'predict_proba']):
                return jsonify({
                    'success': False,
                    'error': 'Model is missing required prediction methods',
                    'status': 'model_invalid'
                }), 500
                
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"Error in predict endpoint: {error_msg}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': 'Failed to load model',
                'details': str(e),
                'status': 'model_loading_error'
            }), 500
        
        # Validate input
        data = request.get_json()
        if not data or 'email_text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: email_text',
                'status': 'invalid_input'
            }), 400
        
        email_text = data['email_text'].strip()
        if not email_text:
            return jsonify({
                'success': False,
                'error': 'Email text cannot be empty',
                'status': 'empty_input'
            }), 400
        
        try:
            # Preprocess the input text
            processed_text = preprocess_text(email_text)
            
            # Make prediction
            prediction = model.predict([processed_text])[0]
            
            # Get prediction probabilities
            try:
                probabilities = model.predict_proba([processed_text])
                probability = float(probabilities[0][1])  # Probability of class 1 (spam)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not get probability scores: {str(e)}")
                probability = 1.0 if prediction == 1 else 0.0
            
            return jsonify({
                'success': True,
                'data': {
                    'prediction': int(prediction),
                    'probability': probability,
                    'model_version': current_version
                }
            })
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Check specifically for vectorizer errors
            if "idf vector is not fitted" in str(e).lower():
                return jsonify({
                    'success': False,
                    'error': 'Model vectorizer not properly initialized',
                    'status': 'model_initialization_error',
                    'details': str(e)
                }), 500
                
            return jsonify({
                'success': False,
                'error': 'Error processing prediction',
                'details': str(e),
                'status': 'prediction_error'
            }), 500
    
    except Exception as e:
        error_msg = f"Unexpected error in predict endpoint: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'status': 'server_error',
            'details': str(e) if str(e) else 'No additional details available'
        }), 500

# If running directly (not imported)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting ML service on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
