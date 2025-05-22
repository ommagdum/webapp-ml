"""Model Management Module

This module provides functionality for managing ML model versions, including version tracking,
model metadata storage, automatic rollback capabilities, and version history management.

The ModelManager class handles the lifecycle of machine learning models, including:
- Maintaining a history of model versions with their accuracy metrics
- Tracking the current active model version
- Implementing automatic model rollback when performance degrades
- Managing model storage with automatic cleanup of old versions

Typical usage example:
    manager = ModelManager(models_dir='path/to/models')
    manager.add_version('v1.0', accuracy=0.85)
    
    # Check if a new model should be rolled back
    if manager.should_rollback(new_accuracy=0.65):
        manager.rollback()
"""

import os
import json
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir='models'):
        """Initialize the ModelManager with a specified models directory.
        
        Args:
            models_dir (str): Directory path where models and metadata will be stored.
                            Defaults to 'models'.
        """
        self.models_dir = os.path.abspath(models_dir)
        self.metadata_file = os.path.join(self.models_dir, 'metadata.json')
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Initializing ModelManager with models directory: {self.models_dir}")
        
        # Initialize metadata
        self.init_metadata()
        
        # Verify the current model can be loaded
        self.verify_model_loading()
        
    def verify_model_loading(self):
        """Verify that the current model can be loaded."""
        current_version = self.load_metadata().get('current_version')
        if current_version:
            model_path = self.get_model_path(current_version)
            try:
                joblib.load(model_path)
                logger.info(f"Successfully verified model at {model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model {current_version} from {model_path}: {str(e)}")
                return False
        return False
        
    def get_model_path(self, version):
        """Get the path to a model file, trying both .joblib and .pkl extensions."""
        # Try .joblib first, then .pkl
        for ext in ['.joblib', '.pkl']:
            path = os.path.join(self.models_dir, f"{version}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No model file found for version {version} in {self.models_dir}")

    def init_metadata(self):
        """Initialize metadata file if it doesn't exist.
        
        Creates a new metadata.json file with empty versions list and
        null current_version if the file doesn't already exist.
        """
        try:
            if not os.path.exists(self.metadata_file):
                metadata = {
                    'versions': [],
                    'current_version': None
                }
                self.save_metadata(metadata)
                logger.info(f"Initialized new metadata file at {self.metadata_file}")
            else:
                # Verify metadata file is valid
                self.load_metadata()
                logger.info(f"Loaded existing metadata from {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error initializing metadata: {str(e)}")
            raise

    def save_metadata(self, metadata):
        """Save metadata to the metadata file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {self.metadata_file}: {str(e)}")
            raise

    def load_metadata(self):
        """Load metadata from the metadata file.
        
        Returns:
            dict: The metadata dictionary containing version history and current version.
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is corrupted
        """
        try:
            if not os.path.exists(self.metadata_file):
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
                
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Validate metadata structure
            if not isinstance(metadata, dict) or 'versions' not in metadata:
                raise ValueError("Invalid metadata format: 'versions' key not found")
                
            return metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata file {self.metadata_file}: {str(e)}")
            # Create a backup of the corrupted file
            backup_file = f"{self.metadata_file}.corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(self.metadata_file, backup_file)
            logger.error(f"Created backup of corrupted metadata file at {backup_file}")
            # Initialize new metadata
            self.init_metadata()
            return self.load_metadata()
        except Exception as e:
            logger.error(f"Unexpected error loading metadata: {str(e)}")
            raise

    def add_version(self, version_id, accuracy):
        """Add a new model version to the version history.
        
        Adds a new model version with its accuracy metric and timestamp.
        Sets the new version as the current version.
        Maintains a rolling window of the last 3 versions, automatically
        removing the oldest version when this limit is exceeded.
        
        Args:
            version_id (str): Unique identifier for the model version.
            accuracy (float): Accuracy metric of the model version.
        """
        metadata = self.load_metadata()
        version_info = {
            'version_id': version_id,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
        }
        
        metadata['versions'].append(version_info)
        # Keep only last 3 versions
        if len(metadata['versions']) > 3:
            old_version = metadata['versions'].pop(0)
            os.remove(os.path.join(self.models_dir, f"{old_version['version_id']}.pkl"))

        metadata['current_version'] = version_id
        self.save_metadata(metadata)

    def should_rollback(self, new_accuracy):
        """Determine if a model should be rolled back based on accuracy drop.
        
        Compares the new accuracy with the accuracy of the current model version.
        Recommends rollback if accuracy drops by more than 15%.
        
        Args:
            new_accuracy (float): Accuracy metric of the new model version.
            
        Returns:
            bool: True if rollback is recommended, False otherwise.
        """
        metadata = self.load_metadata()
        if not metadata['versions']:
            return False
            
        last_version = metadata['versions'][-1]
        accuracy_drop = last_version['accuracy'] - new_accuracy
        return accuracy_drop > 0.15  # 15% threshold

    def rollback(self):
        """Roll back to the previous model version.
        
        Removes the latest version from the version history and
        sets the previous version as the current version.
        
        Returns:
            bool: True if rollback was successful, False if there's no previous version.
        """
        metadata = self.load_metadata()
        if len(metadata['versions']) < 2:
            return False
            
        metadata['versions'].pop()  # Remove latest version
        previous_version = metadata['versions'][-1]
        metadata['current_version'] = previous_version['version_id']
        self.save_metadata(metadata)
        return True
