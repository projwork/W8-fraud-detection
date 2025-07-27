"""
Model evaluation module for fraud detection project.
Handles comprehensive evaluation with metrics appropriate for imbalanced datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score, precision_score, 
    recall_score, accuracy_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class to handle comprehensive model evaluation for fraud detection."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize ModelEvaluator.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.evaluation_results = {}
        
    def evaluate_single_model(self, model: Any, X_test: pd.DataFrame, 
                             y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.
        
        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate curves data
            curves_data = self._calculate_curves(y_test, y_pred_proba)
            
            # Prepare results
            results = {
                'model_name': model_name,
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'curves_data': curves_data,
                'predictions': {
                    'y_pred': y_pred.tolist(),
                    'y_pred_proba': y_pred_proba.tolist(),
                    'y_true': y_test.tolist()
                },
                'evaluation_successful': True,
                'error_message': None
            }
            
            logger.info(f"✅ {model_name} evaluation completed")
            logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}, AUC-PR: {metrics['auc_pr']:.4f}, F1: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}: {str(e)}")
            results = {
                'model_name': model_name,
                'evaluation_successful': False,
                'error_message': str(e)
            }
        
        return results
    
    def evaluate_model_suite(self, models: Dict[str, Any], 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate multiple models on test data.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict: Evaluation results for all models
        """
        logger.info(f"Evaluating {len(models)} models on test set...")
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            results = self.evaluate_single_model(model, X_test, y_test, model_name)
            evaluation_results[model_name] = results
        
        self.evaluation_results = evaluation_results
        
        # Summary
        successful_evals = [name for name, results in evaluation_results.items() 
                          if results.get('evaluation_successful', False)]
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  ✅ Successful: {successful_evals}")
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for imbalanced datasets.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dict: Calculated metrics
        """
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            
            # AUC metrics (important for imbalanced data)
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': average_precision_score(y_true, y_pred_proba),
            
            # Additional metrics
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        }
        
        # Calculate specificity (True Negative Rate)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def _calculate_curves(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """
        Calculate ROC and Precision-Recall curves data.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dict: Curves data
        """
        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        curves_data = {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
        
        return curves_data
    
    def plot_confusion_matrices(self, models_to_plot: List[str] = None) -> None:
        """
        Plot confusion matrices for selected models.
        
        Args:
            models_to_plot: List of model names to plot (None for all)
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available for plotting")
            return
        
        if models_to_plot is None:
            models_to_plot = list(self.evaluation_results.keys())
        
        # Filter successful evaluations
        successful_models = [name for name in models_to_plot 
                           if self.evaluation_results.get(name, {}).get('evaluation_successful', False)]
        
        if not successful_models:
            logger.error("No successful evaluations found for plotting")
            return
        
        n_models = len(successful_models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, model_name in enumerate(successful_models):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_cols > 1 else axes[row]
            
            cm = np.array(self.evaluation_results[model_name]['confusion_matrix'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'])
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Remove empty subplots
        for i in range(len(successful_models), n_rows * n_cols):
            if n_cols > 1:
                row = i // n_cols
                col = i % n_cols
                fig.delaxes(axes[row][col] if n_rows > 1 else axes[col])
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, models_to_plot: List[str] = None) -> None:
        """
        Plot ROC curves for selected models.
        
        Args:
            models_to_plot: List of model names to plot (None for all)
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available for plotting")
            return
        
        if models_to_plot is None:
            models_to_plot = list(self.evaluation_results.keys())
        
        plt.figure(figsize=self.figsize)
        
        for model_name in models_to_plot:
            if not self.evaluation_results.get(model_name, {}).get('evaluation_successful', False):
                continue
                
            curves_data = self.evaluation_results[model_name]['curves_data']
            fpr = curves_data['roc_curve']['fpr']
            tpr = curves_data['roc_curve']['tpr']
            auc_roc = self.evaluation_results[model_name]['metrics']['auc_roc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.3f})', linewidth=2)
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, models_to_plot: List[str] = None) -> None:
        """
        Plot Precision-Recall curves for selected models.
        
        Args:
            models_to_plot: List of model names to plot (None for all)
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available for plotting")
            return
        
        if models_to_plot is None:
            models_to_plot = list(self.evaluation_results.keys())
        
        plt.figure(figsize=self.figsize)
        
        for model_name in models_to_plot:
            if not self.evaluation_results.get(model_name, {}).get('evaluation_successful', False):
                continue
                
            curves_data = self.evaluation_results[model_name]['curves_data']
            precision = curves_data['pr_curve']['precision']
            recall = curves_data['pr_curve']['recall']
            auc_pr = self.evaluation_results[model_name]['metrics']['auc_pr']
            
            plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def create_metrics_comparison(self) -> pd.DataFrame:
        """
        Create a DataFrame comparing metrics across all models.
        
        Returns:
            pd.DataFrame: Metrics comparison table
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            if not results.get('evaluation_successful', False):
                continue
                
            metrics = results['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        if not comparison_data:
            logger.error("No successful evaluations found")
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Round numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        # Sort by AUC-PR (most important metric for imbalanced data)
        if 'auc_pr' in df.columns:
            df = df.sort_values('auc_pr', ascending=False)
        
        return df
    
    def generate_model_recommendation(self, primary_metric: str = 'auc_pr') -> Dict[str, Any]:
        """
        Generate model recommendation based on evaluation results.
        
        Args:
            primary_metric: Primary metric for ranking models
            
        Returns:
            Dict: Recommendation summary
        """
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        # Get metrics comparison
        comparison_df = self.create_metrics_comparison()
        if comparison_df.empty:
            return {"error": "No successful evaluations found"}
        
        # Find best model by primary metric
        if primary_metric not in comparison_df.columns:
            primary_metric = 'auc_pr'  # fallback
        
        best_model_idx = comparison_df[primary_metric].idxmax()
        best_model = comparison_df.loc[best_model_idx]
        
        # Generate recommendation
        recommendation = {
            'recommended_model': best_model['Model'],
            'primary_metric': primary_metric,
            'primary_score': best_model[primary_metric],
            'reasoning': self._generate_reasoning(best_model, comparison_df),
            'performance_summary': {
                'auc_roc': best_model.get('auc_roc', 0),
                'auc_pr': best_model.get('auc_pr', 0),
                'f1_score': best_model.get('f1_score', 0),
                'precision': best_model.get('precision', 0),
                'recall': best_model.get('recall', 0)
            },
            'all_models_ranking': comparison_df[['Model', primary_metric]].to_dict('records')
        }
        
        return recommendation
    
    def _generate_reasoning(self, best_model: pd.Series, comparison_df: pd.DataFrame) -> str:
        """
        Generate reasoning for model recommendation.
        
        Args:
            best_model: Best model row from comparison DataFrame
            comparison_df: Full comparison DataFrame
            
        Returns:
            str: Reasoning text
        """
        model_name = best_model['Model']
        auc_pr = best_model.get('auc_pr', 0)
        auc_roc = best_model.get('auc_roc', 0)
        f1 = best_model.get('f1_score', 0)
        
        reasoning_parts = [
            f"{model_name} is recommended as the best performing model."
        ]
        
        # Performance analysis
        if auc_pr > 0.8:
            reasoning_parts.append("It shows excellent performance on the precision-recall metric (AUC-PR > 0.8), which is crucial for imbalanced fraud detection.")
        elif auc_pr > 0.6:
            reasoning_parts.append("It shows good performance on the precision-recall metric (AUC-PR > 0.6), which is important for imbalanced fraud detection.")
        else:
            reasoning_parts.append("While the precision-recall performance could be improved, it's the best among the evaluated models.")
        
        # Model-specific insights
        if 'logistic' in model_name.lower():
            reasoning_parts.append("Logistic regression provides high interpretability, making it suitable for regulatory environments.")
        elif 'forest' in model_name.lower():
            reasoning_parts.append("Random Forest offers good balance between performance and interpretability with natural feature importance.")
        elif 'lgb' in model_name.lower() or 'lightgbm' in model_name.lower():
            reasoning_parts.append("LightGBM provides excellent performance and is well-suited for fraud detection with its gradient boosting approach.")
        elif 'xgb' in model_name.lower() or 'xgboost' in model_name.lower():
            reasoning_parts.append("XGBoost offers robust performance and handles imbalanced data well with its gradient boosting approach.")
        
        return " ".join(reasoning_parts)
    
    def save_evaluation_results(self, filepath: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            filepath: Path to save results
        """
        import json
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        converted_results = convert_for_json(self.evaluation_results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def print_evaluation_summary(self) -> None:
        """Print a summary of evaluation results."""
        if not self.evaluation_results:
            print("No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        comparison_df = self.create_metrics_comparison()
        if not comparison_df.empty:
            # Display key metrics
            key_metrics = ['Model', 'auc_pr', 'auc_roc', 'f1_score', 'precision', 'recall']
            display_df = comparison_df[key_metrics].copy()
            
            print("\nKey Metrics Comparison:")
            print(display_df.to_string(index=False))
            
            # Best model recommendation
            recommendation = self.generate_model_recommendation()
            if 'recommended_model' in recommendation:
                print(f"\n🏆 RECOMMENDED MODEL: {recommendation['recommended_model']}")
                print(f"   Primary Score (AUC-PR): {recommendation['primary_score']:.4f}")
                print(f"   Reasoning: {recommendation['reasoning']}")
        
        print("\n" + "="*60) 