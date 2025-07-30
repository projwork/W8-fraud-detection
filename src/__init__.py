"""
Fraud Detection Analysis Package

This package provides a comprehensive toolkit for fraud detection analysis,
including data loading, preprocessing, EDA, feature engineering, model building,
training, evaluation, and explainability.
"""

from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor
from .eda_analyzer import EDAAnalyzer
from .feature_engineer import FeatureEngineer
from .imbalance_handler import ImbalanceHandler
from .data_splitter import DataSplitter
from .model_builder import ModelBuilder
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_explainer import ModelExplainer
from .utils import (
    setup_logging, save_processed_data, load_processed_data,
    save_model_artifacts, load_model_artifacts,
    create_feature_summary, plot_feature_importance,
    calculate_fraud_rate_by_feature, compare_distributions,
    memory_usage_optimization, save_experiment_config,
    load_experiment_config
)

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"

__all__ = [
    # Core modules
    'DataLoader',
    'DataPreprocessor', 
    'EDAAnalyzer',
    'FeatureEngineer',
    'ImbalanceHandler',
    
    # Model modules
    'DataSplitter',
    'ModelBuilder',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelExplainer',
    
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
