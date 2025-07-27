"""
Model training module for fraud detection project.
Handles training of ML models with cross-validation and performance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    make_scorer, average_precision_score, f1_score, 
    roc_auc_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle model training and cross-validation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.trained_models = {}
        self.training_results = {}
        self.cv_results = {}
        
    def train_single_model(self, model: Any, X_train: pd.DataFrame, 
                          y_train: pd.Series, model_name: str) -> Dict[str, Any]:
        """
        Train a single model and record training time.
        
        Args:
            model: ML model to train
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Dict: Training results
        """
        logger.info(f"Training {model_name}...")
        
        # Record training start time
        start_time = time.time()
        
        # Train the model
        try:
            model.fit(X_train, y_train)
            training_successful = True
            error_message = None
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            training_successful = False
            error_message = str(e)
        
        # Record training end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store trained model
        if training_successful:
            self.trained_models[model_name] = model
        
        # Prepare results
        results = {
            'model_name': model_name,
            'training_successful': training_successful,
            'training_time': training_time,
            'training_samples': len(X_train),
            'n_features': X_train.shape[1],
            'error_message': error_message
        }
        
        if training_successful:
            logger.info(f"✅ {model_name} trained successfully in {training_time:.2f} seconds")
        else:
            logger.error(f"❌ {model_name} training failed: {error_message}")
        
        return results
    
    def train_model_suite(self, models: Dict[str, Any], 
                         X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict]:
        """
        Train multiple models.
        
        Args:
            models: Dictionary of models to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dict: Training results for all models
        """
        logger.info(f"Training {len(models)} models...")
        
        training_results = {}
        
        for model_name, model in models.items():
            results = self.train_single_model(model, X_train, y_train, model_name)
            training_results[model_name] = results
        
        self.training_results = training_results
        
        # Summary
        successful_models = [name for name, results in training_results.items() 
                           if results['training_successful']]
        failed_models = [name for name, results in training_results.items() 
                        if not results['training_successful']]
        
        logger.info(f"Training completed:")
        logger.info(f"  ✅ Successful: {successful_models}")
        if failed_models:
            logger.info(f"  ❌ Failed: {failed_models}")
        
        return training_results
    
    def cross_validate_models(self, models: Dict[str, Any], 
                             X: pd.DataFrame, y: pd.Series,
                             cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Perform cross-validation for all models.
        
        Args:
            models: Dictionary of trained models
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            
        Returns:
            Dict: Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Define scoring metrics for imbalanced data
        scoring_metrics = {
            'auc_roc': 'roc_auc',
            'auc_pr': make_scorer(average_precision_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score)
        }
        
        # Create stratified K-fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {}
        
        for model_name, model in models.items():
            if model_name not in self.trained_models:
                logger.warning(f"Model {model_name} not found in trained models, skipping CV")
                continue
                
            logger.info(f"Cross-validating {model_name}...")
            model_cv_results = {}
            
            for metric_name, scorer in scoring_metrics.items():
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                    model_cv_results[metric_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'scores': scores.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Error computing {metric_name} for {model_name}: {str(e)}")
                    model_cv_results[metric_name] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'scores': [0.0] * cv_folds
                    }
            
            cv_results[model_name] = model_cv_results
            
            # Log key metrics
            auc_roc = model_cv_results.get('auc_roc', {}).get('mean', 0)
            auc_pr = model_cv_results.get('auc_pr', {}).get('mean', 0)
            f1 = model_cv_results.get('f1', {}).get('mean', 0)
            
            logger.info(f"  {model_name} - AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, F1: {f1:.4f}")
        
        self.cv_results = cv_results
        logger.info("Cross-validation completed")
        
        return cv_results
    
    def train_and_evaluate_suite(self, models: Dict[str, Any],
                                X_train: pd.DataFrame, y_train: pd.Series,
                                cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train models and perform cross-validation in one step.
        
        Args:
            models: Dictionary of models to train
            X_train: Training features
            y_train: Training target
            cv_folds: Number of CV folds
            
        Returns:
            Dict: Combined training and CV results
        """
        # Train models
        training_results = self.train_model_suite(models, X_train, y_train)
        
        # Get successfully trained models
        successful_models = {
            name: models[name] for name, results in training_results.items()
            if results['training_successful']
        }
        
        # Perform cross-validation on successful models
        cv_results = {}
        if successful_models:
            cv_results = self.cross_validate_models(successful_models, X_train, y_train, cv_folds)
        
        # Combine results
        combined_results = {}
        for model_name in models.keys():
            combined_results[model_name] = {
                'training': training_results.get(model_name, {}),
                'cross_validation': cv_results.get(model_name, {})
            }
        
        return combined_results
    
    def get_best_model(self, metric: str = 'auc_pr') -> Tuple[str, Any, float]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for selection ('auc_pr', 'auc_roc', 'f1')
            
        Returns:
            Tuple: (model_name, model_object, score)
        """
        if not self.cv_results:
            logger.error("No cross-validation results available")
            return None, None, 0.0
        
        best_model_name = None
        best_score = -1.0
        
        for model_name, cv_result in self.cv_results.items():
            if metric in cv_result:
                score = cv_result[metric]['mean']
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            best_model = self.trained_models.get(best_model_name)
            logger.info(f"Best model by {metric}: {best_model_name} (score: {best_score:.4f})")
            return best_model_name, best_model, best_score
        else:
            logger.error(f"No valid results found for metric: {metric}")
            return None, None, 0.0
    
    def get_model_rankings(self) -> Dict[str, pd.DataFrame]:
        """
        Get model rankings for all metrics.
        
        Returns:
            Dict: DataFrames with model rankings for each metric
        """
        if not self.cv_results:
            logger.error("No cross-validation results available")
            return {}
        
        metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
        rankings = {}
        
        for metric in metrics:
            model_scores = []
            
            for model_name, cv_result in self.cv_results.items():
                if metric in cv_result:
                    model_scores.append({
                        'model': model_name,
                        'mean_score': cv_result[metric]['mean'],
                        'std_score': cv_result[metric]['std']
                    })
            
            if model_scores:
                df = pd.DataFrame(model_scores)
                df = df.sort_values('mean_score', ascending=False)
                df['rank'] = range(1, len(df) + 1)
                rankings[metric] = df
        
        return rankings
    
    def save_training_results(self, filepath: str) -> None:
        """
        Save training and CV results to file.
        
        Args:
            filepath: Path to save results
        """
        import json
        
        results_to_save = {
            'training_results': self.training_results,
            'cv_results': self.cv_results,
            'model_names': list(self.trained_models.keys())
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the results
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_numpy_types(obj)
        
        converted_results = deep_convert(results_to_save)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        logger.info(f"Training results saved to {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training results.
        
        Returns:
            Dict: Training summary
        """
        if not self.training_results:
            return {"message": "No training results available"}
        
        successful_models = [name for name, results in self.training_results.items() 
                           if results['training_successful']]
        
        total_training_time = sum(results['training_time'] 
                                for results in self.training_results.values()
                                if results['training_successful'])
        
        summary = {
            'total_models': len(self.training_results),
            'successful_models': len(successful_models),
            'failed_models': len(self.training_results) - len(successful_models),
            'total_training_time': total_training_time,
            'average_training_time': total_training_time / len(successful_models) if successful_models else 0,
            'successful_model_names': successful_models
        }
        
        # Add best model for each metric if CV results available
        if self.cv_results:
            metrics = ['auc_roc', 'auc_pr', 'f1']
            for metric in metrics:
                best_name, _, best_score = self.get_best_model(metric)
                if best_name:
                    summary[f'best_{metric}_model'] = {
                        'name': best_name,
                        'score': best_score
                    }
        
        return summary 