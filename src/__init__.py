"""
Fraud Detection Project - Modular Components

This package contains modular components for fraud detection analysis:

Task 1 Modules:
- DataLoader: Load and validate datasets
- DataPreprocessor: Clean and preprocess data
- EDAAnalyzer: Exploratory data analysis
- FeatureEngineer: Create and engineer features
- ImbalanceHandler: Handle class imbalance
- Utils: Utility functions

Task 2 Modules:
- DataSplitter: Data preparation and train-test splits
- ModelBuilder: Build and configure ML models
- ModelTrainer: Train models with cross-validation
- ModelEvaluator: Evaluate models with appropriate metrics
"""

# Task 1 imports
from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor
from .eda_analyzer import EDAAnalyzer
from .feature_engineer import FeatureEngineer
from .imbalance_handler import ImbalanceHandler

# Task 2 imports
from .data_splitter import DataSplitter
from .model_builder import ModelBuilder
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

# Utility imports
from .utils import (
    setup_logging,
    save_processed_data,
    load_processed_data,
    save_model_artifacts,
    load_model_artifacts,
    create_feature_summary,
    plot_feature_importance,
    calculate_fraud_rate_by_feature,
    compare_distributions,
    memory_usage_optimization,
    save_experiment_config,
    load_experiment_config
)

__all__ = [
    # Task 1 classes
    'DataLoader',
    'DataPreprocessor', 
    'EDAAnalyzer',
    'FeatureEngineer',
    'ImbalanceHandler',
    
    # Task 2 classes
    'DataSplitter',
    'ModelBuilder',
    'ModelTrainer',
    'ModelEvaluator',
    
    # Utility functions
    'setup_logging',
    'save_processed_data',
    'load_processed_data',
    'save_model_artifacts',
    'load_model_artifacts',
    'create_feature_summary',
    'plot_feature_importance',
    'calculate_fraud_rate_by_feature',
    'compare_distributions',
    'memory_usage_optimization',
    'save_experiment_config',
    'load_experiment_config'
]

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"
__description__ = "Modular fraud detection analysis framework"
