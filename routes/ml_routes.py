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
    """Get the current model from the global variable in app.py"""
    from app import model
    return model

@ml_routes.route('/retrain', methods=['POST'])
def retrain():
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
