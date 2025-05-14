"""
Shared ML Resources Module

This module provides shared resources for the ML components of the application.
It initializes and exports singleton instances of key components that need to be
accessed across multiple modules, avoiding circular imports and redundant initialization.

Resources:
    model_manager: A singleton ModelManager instance that handles model loading,
                  caching, and management across the application.
"""

from ml.management.model_manager import ModelManager

# Shared resources
model_manager = ModelManager(models_dir='models')
