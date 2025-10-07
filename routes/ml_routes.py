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
    """Get the current model from the global model manager."""
    global model_manager
    try:
        metadata = model_manager.load_metadata()
        current_version = metadata.get('current_version', 'initial_model')
        model_path = model_manager.get_model_path(current_version)
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading current model: {str(e)}")
        return None


@ml_routes.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model with new training data."""
    try:
        data = request.get_json()
        
        # Handle Spring Boot format: {"trainingData": [{"content": "...", "label": ...}]}
        if isinstance(data, dict) and 'trainingData' in data:
            training_data = data['trainingData']
            # Convert 'content' to 'text' for the ML service
            for item in training_data:
                if 'content' in item and 'text' not in item:
                    item['text'] = item['content']
        else:
            # Direct format (fallback)
            training_data = data
        
        new_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_manager.models_dir, f"{new_version}.joblib")
        
        current_model = get_current_model()
        if current_model is None:
            return jsonify({
                "success": False,
                "error": "Current model not available for retraining"
            }), 500
            
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
