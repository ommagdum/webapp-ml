import os
import json
from datetime import datetime

class ModelManager:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, 'metadata.json')
        self.init_metadata()

    def init_metadata(self):
        if not os.path.exists(self.metadata_file):
            metadata = {
                'versions': [],
                'current_version': None
            }
            self.save_metadata(metadata)

    def save_metadata(self, metadata):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self):
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def add_version(self, version_id, accuracy):
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
        metadata = self.load_metadata()
        if not metadata['versions']:
            return False
            
        last_version = metadata['versions'][-1]
        accuracy_drop = last_version['accuracy'] - new_accuracy
        return accuracy_drop > 0.15  # 15% threshold

    def rollback(self):
        metadata = self.load_metadata()
        if len(metadata['versions']) < 2:
            return False
            
        metadata['versions'].pop()  # Remove latest version
        previous_version = metadata['versions'][-1]
        metadata['current_version'] = previous_version['version_id']
        self.save_metadata(metadata)
        return True
