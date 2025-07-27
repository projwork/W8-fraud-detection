"""
Model building module for fraud detection project.
Handles creation and configuration of ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')

# Try to import gradient boosting models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

logger = logging.getLogger(__name__)

class ModelBuilder:
    """Class to build and configure ML models for fraud detection."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelBuilder.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.model_configs = {}
        
    def create_logistic_regression(self, **kwargs) -> LogisticRegression:
        """
        Create Logistic Regression model.
        
        Args:
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            LogisticRegression: Configured model
        """
        default_params = {
            'random_state': self.random_state,
            'max_iter': 1000,
            'class_weight': 'balanced',  # Handle class imbalance
            'solver': 'liblinear'  # Good for small datasets
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        
        self.model_configs['logistic_regression'] = default_params
        logger.info(f"Created Logistic Regression with params: {default_params}")
        
        return model
    
    def create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """
        Create Random Forest model.
        
        Args:
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            RandomForestClassifier: Configured model
        """
        default_params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1  # Use all available cores
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        
        self.model_configs['random_forest'] = default_params
        logger.info(f"Created Random Forest with params: {default_params}")
        
        return model
    
    def create_lightgbm(self, **kwargs) -> Optional[Any]:
        """
        Create LightGBM model.
        
        Args:
            **kwargs: Additional parameters for LGBMClassifier
            
        Returns:
            LGBMClassifier or None: Configured model if available
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available. Install with: pip install lightgbm")
            return None
        
        default_params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        model = lgb.LGBMClassifier(**default_params)
        
        self.model_configs['lightgbm'] = default_params
        logger.info(f"Created LightGBM with params: {default_params}")
        
        return model
    
    def create_xgboost(self, **kwargs) -> Optional[Any]:
        """
        Create XGBoost model.
        
        Args:
            **kwargs: Additional parameters for XGBClassifier
            
        Returns:
            XGBClassifier or None: Configured model if available
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. Install with: pip install xgboost")
            return None
        
        default_params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_jobs': -1
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        
        self.model_configs['xgboost'] = default_params
        logger.info(f"Created XGBoost with params: {default_params}")
        
        return model
    
    def create_model_suite(self, include_models: List[str] = None) -> Dict[str, Any]:
        """
        Create a suite of models for comparison.
        
        Args:
            include_models: List of model names to include
            
        Returns:
            Dict: Dictionary of model names and instances
        """
        if include_models is None:
            include_models = ['logistic_regression', 'random_forest']
            
            # Add gradient boosting models if available
            if LIGHTGBM_AVAILABLE:
                include_models.append('lightgbm')
            elif XGBOOST_AVAILABLE:
                include_models.append('xgboost')
        
        models = {}
        
        for model_name in include_models:
            if model_name == 'logistic_regression':
                models[model_name] = self.create_logistic_regression()
            elif model_name == 'random_forest':
                models[model_name] = self.create_random_forest()
            elif model_name == 'lightgbm':
                model = self.create_lightgbm()
                if model is not None:
                    models[model_name] = model
            elif model_name == 'xgboost':
                model = self.create_xgboost()
                if model is not None:
                    models[model_name] = model
            else:
                logger.warning(f"Unknown model name: {model_name}")
        
        self.models = models
        logger.info(f"Created model suite with {len(models)} models: {list(models.keys())}")
        
        return models
    
    def optimize_for_imbalanced_data(self, models: Dict[str, Any], 
                                   imbalance_ratio: float) -> Dict[str, Any]:
        """
        Optimize model parameters for imbalanced datasets.
        
        Args:
            models: Dictionary of models
            imbalance_ratio: Class imbalance ratio (minority/majority)
            
        Returns:
            Dict: Optimized models
        """
        optimized_models = {}
        
        for model_name, model in models.items():
            if model_name == 'logistic_regression':
                # Adjust class weights based on imbalance
                if imbalance_ratio < 0.1:  # Severe imbalance
                    model.set_params(class_weight='balanced', C=0.1)
                else:
                    model.set_params(class_weight='balanced', C=1.0)
                    
            elif model_name == 'random_forest':
                # Adjust parameters for imbalanced data
                if imbalance_ratio < 0.1:
                    model.set_params(
                        class_weight='balanced_subsample',
                        min_samples_leaf=2,
                        max_depth=10
                    )
                else:
                    model.set_params(class_weight='balanced')
                    
            elif model_name == 'lightgbm':
                # Calculate scale_pos_weight for LightGBM
                scale_pos_weight = (1 - imbalance_ratio) / imbalance_ratio
                model.set_params(scale_pos_weight=scale_pos_weight)
                
            elif model_name == 'xgboost':
                # Calculate scale_pos_weight for XGBoost
                scale_pos_weight = (1 - imbalance_ratio) / imbalance_ratio
                model.set_params(scale_pos_weight=scale_pos_weight)
            
            optimized_models[model_name] = model
        
        logger.info(f"Optimized {len(optimized_models)} models for imbalanced data")
        return optimized_models
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get information about created models.
        
        Returns:
            Dict: Model information and configurations
        """
        info = {}
        
        for model_name, config in self.model_configs.items():
            info[model_name] = {
                'type': model_name,
                'parameters': config,
                'suitable_for': self._get_model_characteristics(model_name)
            }
        
        return info
    
    def _get_model_characteristics(self, model_name: str) -> Dict[str, str]:
        """
        Get characteristics of different models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Model characteristics
        """
        characteristics = {
            'logistic_regression': {
                'interpretability': 'High',
                'training_speed': 'Fast',
                'prediction_speed': 'Very Fast',
                'memory_usage': 'Low',
                'handling_imbalance': 'Good with class_weight',
                'best_for': 'Baseline, interpretable models'
            },
            'random_forest': {
                'interpretability': 'Medium',
                'training_speed': 'Medium',
                'prediction_speed': 'Fast',
                'memory_usage': 'Medium',
                'handling_imbalance': 'Good with class_weight',
                'best_for': 'Robust performance, feature importance'
            },
            'lightgbm': {
                'interpretability': 'Low',
                'training_speed': 'Fast',
                'prediction_speed': 'Fast',
                'memory_usage': 'Low',
                'handling_imbalance': 'Excellent',
                'best_for': 'High performance, large datasets'
            },
            'xgboost': {
                'interpretability': 'Low',
                'training_speed': 'Medium',
                'prediction_speed': 'Fast',
                'memory_usage': 'Medium',
                'handling_imbalance': 'Excellent',
                'best_for': 'High performance, competitions'
            }
        }
        
        return characteristics.get(model_name, {})
    
    def get_recommended_models(self, dataset_size: int, 
                             imbalance_ratio: float) -> List[str]:
        """
        Get recommended models based on dataset characteristics.
        
        Args:
            dataset_size: Number of samples in dataset
            imbalance_ratio: Class imbalance ratio
            
        Returns:
            List: Recommended model names
        """
        recommendations = ['logistic_regression']  # Always include baseline
        
        if dataset_size < 1000:
            # Small dataset
            recommendations.append('random_forest')
        elif dataset_size < 100000:
            # Medium dataset
            recommendations.extend(['random_forest'])
            if LIGHTGBM_AVAILABLE:
                recommendations.append('lightgbm')
            elif XGBOOST_AVAILABLE:
                recommendations.append('xgboost')
        else:
            # Large dataset
            if LIGHTGBM_AVAILABLE:
                recommendations.append('lightgbm')
            if XGBOOST_AVAILABLE:
                recommendations.append('xgboost')
            recommendations.append('random_forest')
        
        logger.info(f"Recommended models for dataset (size={dataset_size}, "
                   f"imbalance={imbalance_ratio:.4f}): {recommendations}")
        
        return recommendations 