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
from datetime import datetime

class ModelManager:
    def __init__(self, models_dir='models'):
        """Initialize the ModelManager with a specified models directory.
        
        Args:
            models_dir (str): Directory path where models and metadata will be stored.
                              Defaults to 'models'.
        """
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, 'metadata.json')
        self.init_metadata()

    def init_metadata(self):
        """Initialize metadata file if it doesn't exist.
        
        Creates a new metadata.json file with empty versions list and
        null current_version if the file doesn't already exist.
        """
        if not os.path.exists(self.metadata_file):
            metadata = {
                'versions': [],
                'current_version': None
            }
            self.save_metadata(metadata)

    def save_metadata(self, metadata):
        """Save metadata to the metadata file.
        
        Args:
            metadata (dict): The metadata dictionary to save.
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self):
        """Load metadata from the metadata file.
        
        Returns:
            dict: The metadata dictionary containing version history and current version.
        """
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

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
