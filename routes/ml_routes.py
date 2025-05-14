"""ML Routes Module

This module defines Flask Blueprint routes for ML model management operations.
It provides API endpoints for model retraining and integrates with the model
management system for version control and automatic rollback capabilities.

The module handles:
- Model retraining with new data
- Automatic model quality assessment
- Rollback to previous model versions when quality degrades
- Version tracking and management

These routes are designed to be mounted to a Flask application with a
prefix (typically '/ml') to create a dedicated API namespace for ML operations.
"""

from ml.management.model_manager import ModelManager
from flask import Blueprint, request, jsonify
from datetime import datetime
from ml.trainer import retrain_model
import joblib
import os
from ml.shared import model_manager


# Use the same models_dir as in app.py
model_manager = ModelManager(models_dir='models')

ml_routes = Blueprint('ml_routes', __name__)

def get_current_model():
    """Get the current model from the global variable in app.py.
    
    This function imports the model from app.py to avoid circular imports.
    It provides access to the currently loaded model instance that is
    being used for predictions.
    
    Returns:
        object: The currently loaded ML model instance from app.py
    """
    from app import model
    return model

@ml_routes.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model with new training data.
    
    This endpoint accepts new training data via a POST request and uses it to
    retrain the current model. It implements an automatic quality control system
    that rolls back to the previous model version if the accuracy drops significantly.
    
    The function:
    1. Receives new training data in JSON format
    2. Creates a new model version with timestamp-based ID
    3. Retrains the model with the new data
    4. Evaluates the new model's accuracy
    5. Automatically rolls back if accuracy drops below threshold
    6. Otherwise, registers the new model version
    
    Request Body:
        JSON array of training samples, each with 'text' and 'label' fields
        
    Returns:
        JSON: A response containing:
            - success: Boolean indicating if retraining was successful
            - new_version: ID of the new model version (if successful)
            - accuracy: Accuracy score of the new model
            - message: Explanation if model was rolled back
            - error: Error message if an exception occurred
            
    Response Codes:
        - 200: Successful retraining or controlled rollback
        - 500: Server error during processing
    """
    try:
        training_data = request.get_json()
        new_version = datetime.now().strftime('%Y%m%d_%H%M')
        model_path = os.path.join(model_manager.models_dir, f"{new_version}.pkl")
        
        current_model = get_current_model()
        accuracy = retrain_model(current_model, training_data, model_path)
        
        if model_manager.should_rollback(accuracy):
            model_manager.rollback()
            return jsonify({
                "success": False,
                "message": "Model rolled back due to accuracy drop",
                "accuracy": accuracy
            })
        
        model_manager.add_version(new_version, accuracy)
        
        return jsonify({
            "success": True,
            "new_version": new_version,
            "accuracy": accuracy
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
