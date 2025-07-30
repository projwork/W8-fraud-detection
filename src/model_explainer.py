"""
Model explainability module using SHAP for fraud detection project.
Provides comprehensive model interpretation and visualization capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Class to provide SHAP-based model explainability for fraud detection models."""
    
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, 
                 X_test: pd.DataFrame, feature_names: List[str] = None):
        """
        Initialize ModelExplainer.
        
        Args:
            model: Trained model to explain
            X_train: Training features for SHAP explainer
            X_test: Test features for explanation
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available. Install with: pip install shap")
        
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None)
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values = None
        self.shap_values_test = None
        
        # Set up explainer based on model type
        self._setup_explainer()
        
    def _setup_explainer(self):
        """Setup appropriate SHAP explainer based on model type."""
        try:
            if isinstance(self.model, (RandomForestClassifier,)):
                # Tree explainer for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer for tree-based model")
                
            elif isinstance(self.model, LogisticRegression):
                # Linear explainer for linear models
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                logger.info("Using LinearExplainer for linear model")
                
            else:
                # Kernel explainer as fallback (slower but works for any model)
                # Use a sample for background to speed up computation
                background_size = min(100, len(self.X_train))
                background = shap.sample(self.X_train, background_size)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                logger.info(f"Using KernelExplainer with background size {background_size}")
                
        except Exception as e:
            logger.warning(f"Failed to create specialized explainer: {e}")
            # Fallback to kernel explainer
            background_size = min(50, len(self.X_train))
            background = shap.sample(self.X_train, background_size)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            logger.info("Using KernelExplainer as fallback")
    
    def calculate_shap_values(self, sample_size: int = None, on_test: bool = True) -> np.ndarray:
        """
        Calculate SHAP values for the dataset.
        
        Args:
            sample_size: Number of samples to calculate SHAP for (None for all)
            on_test: Whether to calculate on test set (True) or train set (False)
            
        Returns:
            np.ndarray: SHAP values
        """
        dataset = self.X_test if on_test else self.X_train
        
        if sample_size and len(dataset) > sample_size:
            # Sample for faster computation
            sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
            data_sample = dataset.iloc[sample_indices] if hasattr(dataset, 'iloc') else dataset[sample_indices]
        else:
            data_sample = dataset
        
        logger.info(f"Calculating SHAP values for {len(data_sample)} samples...")
        
        try:
            if isinstance(self.model, LogisticRegression):
                # For linear models, get SHAP values directly
                shap_values = self.explainer.shap_values(data_sample)
            else:
                # For tree models and others
                shap_values = self.explainer.shap_values(data_sample)
                
                # Handle binary classification case
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # Take positive class SHAP values
                    shap_values = shap_values[1]
                    
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            raise
        
        if on_test:
            self.shap_values_test = shap_values
        else:
            self.shap_values = shap_values
            
        logger.info("SHAP values calculated successfully")
        return shap_values
    
    def plot_summary(self, shap_values: np.ndarray = None, 
                    dataset: pd.DataFrame = None, plot_type: str = 'dot',
                    max_display: int = 20, title: str = None) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values to plot (uses calculated if None)
            dataset: Dataset for feature values (uses test if None)
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
            title: Custom title for the plot
        """
        if shap_values is None:
            if self.shap_values_test is None:
                shap_values = self.calculate_shap_values()
            else:
                shap_values = self.shap_values_test
                
        if dataset is None:
            dataset = self.X_test
            
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(shap_values, dataset, 
                            feature_names=self.feature_names,
                            plot_type='bar', max_display=max_display,
                            show=False)
        else:
            shap.summary_plot(shap_values, dataset,
                            feature_names=self.feature_names,
                            max_display=max_display, show=False)
        
        if title:
            plt.title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_force_plot(self, instance_idx: int = 0, 
                       shap_values: np.ndarray = None,
                       dataset: pd.DataFrame = None,
                       matplotlib: bool = True) -> None:
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            instance_idx: Index of instance to explain
            shap_values: SHAP values (uses calculated if None)
            dataset: Dataset (uses test if None)
            matplotlib: Whether to use matplotlib backend
        """
        if shap_values is None:
            if self.shap_values_test is None:
                shap_values = self.calculate_shap_values()
            else:
                shap_values = self.shap_values_test
                
        if dataset is None:
            dataset = self.X_test
            
        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]
        else:
            expected_value = 0
        
        if matplotlib:
            plt.figure(figsize=(16, 4))
            shap.force_plot(expected_value, 
                           shap_values[instance_idx],
                           dataset.iloc[instance_idx] if hasattr(dataset, 'iloc') else dataset[instance_idx],
                           feature_names=self.feature_names,
                           matplotlib=True, show=False)
            plt.tight_layout()
            plt.show()
        else:
            # Interactive plot (Jupyter notebook)
            return shap.force_plot(expected_value, 
                                 shap_values[instance_idx],
                                 dataset.iloc[instance_idx] if hasattr(dataset, 'iloc') else dataset[instance_idx],
                                 feature_names=self.feature_names)
    
    def plot_waterfall(self, instance_idx: int = 0,
                      shap_values: np.ndarray = None,
                      dataset: pd.DataFrame = None) -> None:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            instance_idx: Index of instance to explain
            shap_values: SHAP values (uses calculated if None)
            dataset: Dataset (uses test if None)
        """
        if shap_values is None:
            if self.shap_values_test is None:
                shap_values = self.calculate_shap_values()
            else:
                shap_values = self.shap_values_test
                
        if dataset is None:
            dataset = self.X_test
            
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]
        else:
            expected_value = 0
        
        plt.figure(figsize=(12, 8))
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=expected_value,
            data=dataset.iloc[instance_idx] if hasattr(dataset, 'iloc') else dataset[instance_idx],
            feature_names=self.feature_names
        )
        
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.show()
    
    def plot_dependence(self, feature: Union[str, int], 
                       interaction_feature: Union[str, int, None] = None,
                       shap_values: np.ndarray = None,
                       dataset: pd.DataFrame = None) -> None:
        """
        Create SHAP dependence plot.
        
        Args:
            feature: Feature to plot dependence for
            interaction_feature: Feature to color by for interactions
            shap_values: SHAP values (uses calculated if None)
            dataset: Dataset (uses test if None)
        """
        if shap_values is None:
            if self.shap_values_test is None:
                shap_values = self.calculate_shap_values()
            else:
                shap_values = self.shap_values_test
                
        if dataset is None:
            dataset = self.X_test
            
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, dataset,
                           feature_names=self.feature_names,
                           interaction_index=interaction_feature,
                           show=False)
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, shap_values: np.ndarray = None,
                              method: str = 'mean_abs') -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values (uses calculated if None)
            method: Method to calculate importance ('mean_abs', 'mean', 'std')
            
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if shap_values is None:
            if self.shap_values_test is None:
                shap_values = self.calculate_shap_values()
            else:
                shap_values = self.shap_values_test
        
        feature_names = self.feature_names or [f'Feature_{i}' for i in range(shap_values.shape[1])]
        
        if method == 'mean_abs':
            importance = np.abs(shap_values).mean(axis=0)
        elif method == 'mean':
            importance = shap_values.mean(axis=0)
        elif method == 'std':
            importance = shap_values.std(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False, key=abs)
        
        return importance_df
    
    def analyze_fraud_drivers(self, shap_values: np.ndarray = None,
                             y_test: pd.Series = None,
                             top_features: int = 10) -> Dict[str, Any]:
        """
        Analyze key drivers of fraud predictions.
        
        Args:
            shap_values: SHAP values (uses calculated if None)
            y_test: True labels for test set
            top_features: Number of top features to analyze
            
        Returns:
            Dict: Analysis results including key insights
        """
        if shap_values is None:
            if self.shap_values_test is None:
                shap_values = self.calculate_shap_values()
            else:
                shap_values = self.shap_values_test
        
        # Get feature importance
        importance_df = self.get_feature_importance(shap_values)
        top_important_features = importance_df.head(top_features)
        
        analysis = {
            'top_features': top_important_features,
            'overall_insights': [],
            'fraud_specific_insights': [],
            'feature_statistics': {}
        }
        
        # Overall feature importance insights
        analysis['overall_insights'].append(
            f"The most important feature for fraud detection is '{top_important_features.iloc[0]['feature']}'"
        )
        
        # Analyze positive vs negative contributions
        for idx, row in top_important_features.iterrows():
            feature_name = row['feature']
            if hasattr(self.X_test, 'columns'):
                feature_idx = self.X_test.columns.get_loc(feature_name)
            else:
                feature_idx = list(self.feature_names).index(feature_name)
            
            feature_shap = shap_values[:, feature_idx]
            
            analysis['feature_statistics'][feature_name] = {
                'mean_shap': float(np.mean(feature_shap)),
                'std_shap': float(np.std(feature_shap)),
                'positive_contributions': float(np.sum(feature_shap > 0)),
                'negative_contributions': float(np.sum(feature_shap < 0)),
                'max_positive_impact': float(np.max(feature_shap)),
                'max_negative_impact': float(np.min(feature_shap))
            }
        
        # Fraud-specific insights if labels are provided
        if y_test is not None:
            fraud_indices = y_test == 1
            non_fraud_indices = y_test == 0
            
            if np.sum(fraud_indices) > 0:
                fraud_shap_mean = np.mean(shap_values[fraud_indices], axis=0)
                non_fraud_shap_mean = np.mean(shap_values[non_fraud_indices], axis=0)
                
                # Find features that contribute most to fraud vs non-fraud
                fraud_diff = fraud_shap_mean - non_fraud_shap_mean
                feature_names = self.feature_names or [f'Feature_{i}' for i in range(len(fraud_diff))]
                
                fraud_drivers = pd.DataFrame({
                    'feature': feature_names,
                    'fraud_contribution_diff': fraud_diff
                }).sort_values('fraud_contribution_diff', ascending=False, key=abs)
                
                analysis['fraud_drivers'] = fraud_drivers.head(top_features)
                
                # Generate insights
                top_fraud_driver = fraud_drivers.iloc[0]
                analysis['fraud_specific_insights'].append(
                    f"'{top_fraud_driver['feature']}' shows the largest difference in SHAP contributions "
                    f"between fraud and non-fraud cases ({top_fraud_driver['fraud_contribution_diff']:.4f})"
                )
        
        return analysis
    
    def generate_explanation_report(self, y_test: pd.Series = None,
                                  dataset_name: str = "Dataset") -> str:
        """
        Generate a comprehensive explanation report.
        
        Args:
            y_test: True labels for test set
            dataset_name: Name of the dataset being analyzed
            
        Returns:
            str: Formatted explanation report
        """
        if self.shap_values_test is None:
            self.calculate_shap_values()
        
        analysis = self.analyze_fraud_drivers(y_test=y_test)
        
        report = f"""
# SHAP Model Explainability Report - {dataset_name}

## Model Overview
- Model Type: {type(self.model).__name__}
- Features Analyzed: {len(self.feature_names) if self.feature_names else 'Unknown'}
- Test Samples: {len(self.X_test)}

## Top Feature Importance Rankings
"""
        
        for idx, row in analysis['top_features'].head(10).iterrows():
            report += f"{idx+1}. {row['feature']}: {row['importance']:.4f}\n"
        
        report += "\n## Key Insights\n"
        for insight in analysis['overall_insights']:
            report += f"- {insight}\n"
        
        if 'fraud_specific_insights' in analysis:
            report += "\n## Fraud-Specific Insights\n"
            for insight in analysis['fraud_specific_insights']:
                report += f"- {insight}\n"
        
        if 'fraud_drivers' in analysis:
            report += "\n## Top Fraud Drivers (vs Non-Fraud)\n"
            for idx, row in analysis['fraud_drivers'].head(5).iterrows():
                direction = "increases" if row['fraud_contribution_diff'] > 0 else "decreases"
                report += f"- {row['feature']}: {direction} fraud probability (diff: {row['fraud_contribution_diff']:.4f})\n"
        
        return report 