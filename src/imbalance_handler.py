"""
Class imbalance handling module for fraud detection project.
Implements various sampling techniques to handle imbalanced datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
from sklearn.model_selection import train_test_split
from collections import Counter

# Sampling techniques
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("imbalanced-learn not available. Some sampling techniques will not work.")

logger = logging.getLogger(__name__)

class ImbalanceHandler:
    """Class to handle class imbalance in datasets."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ImbalanceHandler.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.samplers = {}
        
    def analyze_class_distribution(self, y: pd.Series) -> Dict:
        """
        Analyze the class distribution in the target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Dict: Class distribution analysis
        """
        class_counts = Counter(y)
        total_samples = len(y)
        
        distribution = {
            'class_counts': dict(class_counts),
            'class_percentages': {k: (v/total_samples)*100 for k, v in class_counts.items()},
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'majority_class': max(class_counts, key=class_counts.get),
            'minority_class': min(class_counts, key=class_counts.get),
            'imbalance_ratio': min(class_counts.values()) / max(class_counts.values())
        }
        
        # Determine imbalance severity
        if distribution['imbalance_ratio'] < 0.1:
            distribution['severity'] = 'Severe'
        elif distribution['imbalance_ratio'] < 0.3:
            distribution['severity'] = 'Moderate'
        else:
            distribution['severity'] = 'Mild'
            
        logger.info(f"Class distribution analysis completed. Imbalance ratio: {distribution['imbalance_ratio']:.4f}")
        return distribution
    
    def simple_random_oversample(self, X: pd.DataFrame, y: pd.Series, 
                                strategy: Union[str, Dict] = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simple random oversampling of minority class.
        
        Args:
            X: Feature matrix
            y: Target variable
            strategy: Sampling strategy ('auto', 'minority', or dict)
            
        Returns:
            Tuple: Resampled X and y
        """
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        class_counts = Counter(y)
        majority_count = max(class_counts.values())
        
        if strategy == 'auto':
            target_counts = {cls: majority_count for cls in class_counts.keys()}
        elif strategy == 'minority':
            minority_class = min(class_counts, key=class_counts.get)
            target_counts = {minority_class: majority_count}
        else:
            target_counts = strategy
            
        for class_label, target_count in target_counts.items():
            current_count = class_counts[class_label]
            if target_count > current_count:
                # Get samples of this class
                class_samples = X[y == class_label]
                samples_needed = target_count - current_count
                
                # Random oversampling
                oversample_indices = np.random.choice(
                    class_samples.index, 
                    size=samples_needed, 
                    replace=True
                )
                
                # Add oversampled data
                X_resampled = pd.concat([X_resampled, X.loc[oversample_indices]])
                y_resampled = pd.concat([y_resampled, y.loc[oversample_indices]])
                
        logger.info(f"Simple random oversampling completed. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def simple_random_undersample(self, X: pd.DataFrame, y: pd.Series,
                                 strategy: Union[str, Dict] = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simple random undersampling of majority class.
        
        Args:
            X: Feature matrix
            y: Target variable
            strategy: Sampling strategy ('auto', 'majority', or dict)
            
        Returns:
            Tuple: Resampled X and y
        """
        class_counts = Counter(y)
        minority_count = min(class_counts.values())
        
        if strategy == 'auto':
            target_counts = {cls: minority_count for cls in class_counts.keys()}
        elif strategy == 'majority':
            majority_class = max(class_counts, key=class_counts.get)
            target_counts = {majority_class: minority_count}
        else:
            target_counts = strategy
            
        indices_to_keep = []
        
        for class_label in class_counts.keys():
            class_indices = X[y == class_label].index
            target_count = target_counts.get(class_label, len(class_indices))
            
            if target_count < len(class_indices):
                # Random undersampling
                selected_indices = np.random.choice(
                    class_indices, 
                    size=target_count, 
                    replace=False
                )
                indices_to_keep.extend(selected_indices)
            else:
                indices_to_keep.extend(class_indices)
                
        X_resampled = X.loc[indices_to_keep]
        y_resampled = y.loc[indices_to_keep]
        
        logger.info(f"Simple random undersampling completed. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series, 
                   variant: str = 'standard', **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique) variants.
        
        Args:
            X: Feature matrix
            y: Target variable
            variant: SMOTE variant ('standard', 'borderline', 'svm', 'adasyn')
            **kwargs: Additional parameters for SMOTE
            
        Returns:
            Tuple: Resampled X and y
        """
        if not IMBLEARN_AVAILABLE:
            logger.error("imbalanced-learn not available. Install with: pip install imbalanced-learn")
            return X, y
            
        # Select SMOTE variant
        if variant == 'standard':
            sampler = SMOTE(random_state=self.random_state, **kwargs)
        elif variant == 'borderline':
            sampler = BorderlineSMOTE(random_state=self.random_state, **kwargs)
        elif variant == 'svm':
            sampler = SVMSMOTE(random_state=self.random_state, **kwargs)
        elif variant == 'adasyn':
            sampler = ADASYN(random_state=self.random_state, **kwargs)
        else:
            raise ValueError(f"Unknown SMOTE variant: {variant}")
            
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series with proper column names
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            logger.info(f"SMOTE ({variant}) completed. New shape: {X_resampled.shape}")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            logger.info("Falling back to simple random oversampling")
            return self.simple_random_oversample(X, y)
    
    def apply_undersampling(self, X: pd.DataFrame, y: pd.Series,
                           method: str = 'random', **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply undersampling techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Undersampling method ('random', 'tomek', 'enn')
            **kwargs: Additional parameters
            
        Returns:
            Tuple: Resampled X and y
        """
        if not IMBLEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available. Using simple random undersampling")
            return self.simple_random_undersample(X, y)
            
        try:
            if method == 'random':
                sampler = RandomUnderSampler(random_state=self.random_state, **kwargs)
            elif method == 'tomek':
                sampler = TomekLinks(**kwargs)
            elif method == 'enn':
                sampler = EditedNearestNeighbours(**kwargs)
            else:
                raise ValueError(f"Unknown undersampling method: {method}")
                
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            logger.info(f"Undersampling ({method}) completed. New shape: {X_resampled.shape}")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Undersampling failed: {e}")
            logger.info("Falling back to simple random undersampling")
            return self.simple_random_undersample(X, y)
    
    def apply_combined_sampling(self, X: pd.DataFrame, y: pd.Series,
                               method: str = 'smote_tomek', **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply combined over- and under-sampling techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Combined method ('smote_tomek', 'smote_enn')
            **kwargs: Additional parameters
            
        Returns:
            Tuple: Resampled X and y
        """
        if not IMBLEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available. Using simple random sampling")
            # Combine simple over and undersampling
            X_over, y_over = self.simple_random_oversample(X, y)
            return self.simple_random_undersample(X_over, y_over, strategy='majority')
            
        try:
            if method == 'smote_tomek':
                sampler = SMOTETomek(random_state=self.random_state, **kwargs)
            elif method == 'smote_enn':
                sampler = SMOTEENN(random_state=self.random_state, **kwargs)
            else:
                raise ValueError(f"Unknown combined sampling method: {method}")
                
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            logger.info(f"Combined sampling ({method}) completed. New shape: {X_resampled.shape}")
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Combined sampling failed: {e}")
            logger.info("Falling back to simple combined sampling")
            X_over, y_over = self.simple_random_oversample(X, y)
            return self.simple_random_undersample(X_over, y_over, strategy='majority')
    
    def recommend_sampling_strategy(self, y: pd.Series, dataset_size: str = 'medium') -> str:
        """
        Recommend sampling strategy based on imbalance ratio and dataset size.
        
        Args:
            y: Target variable
            dataset_size: Size of dataset ('small', 'medium', 'large')
            
        Returns:
            str: Recommended sampling strategy
        """
        distribution = self.analyze_class_distribution(y)
        imbalance_ratio = distribution['imbalance_ratio']
        total_samples = distribution['total_samples']
        
        # Size-based recommendations
        if total_samples < 1000:
            actual_size = 'small'
        elif total_samples < 100000:
            actual_size = 'medium'
        else:
            actual_size = 'large'
            
        # Recommendation logic
        if imbalance_ratio >= 0.3:  # Mild imbalance
            recommendation = "No sampling needed - use class weights instead"
        elif imbalance_ratio >= 0.1:  # Moderate imbalance
            if actual_size == 'small':
                recommendation = "SMOTE or simple oversampling"
            else:
                recommendation = "SMOTE or combined SMOTE-Tomek"
        else:  # Severe imbalance
            if actual_size == 'small':
                recommendation = "SMOTE (be careful of overfitting)"
            elif actual_size == 'medium':
                recommendation = "Combined SMOTE-Tomek or SMOTE-ENN"
            else:
                recommendation = "Random undersampling or Tomek links"
                
        logger.info(f"Recommendation for imbalance ratio {imbalance_ratio:.4f}: {recommendation}")
        return recommendation
    
    def create_balanced_train_test_split(self, X: pd.DataFrame, y: pd.Series,
                                       test_size: float = 0.2, 
                                       sampling_method: str = 'smote',
                                       **sampling_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train-test split and apply sampling only to training data.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set
            sampling_method: Sampling method to apply
            **sampling_kwargs: Additional sampling parameters
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test (with balanced training set)
        """
        # First split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Original train set shape: {X_train.shape}")
        logger.info(f"Original train class distribution: {Counter(y_train)}")
        
        # Apply sampling only to training data
        if sampling_method == 'smote':
            X_train_balanced, y_train_balanced = self.apply_smote(X_train, y_train, **sampling_kwargs)
        elif sampling_method == 'undersample':
            X_train_balanced, y_train_balanced = self.apply_undersampling(X_train, y_train, **sampling_kwargs)
        elif sampling_method == 'combined':
            X_train_balanced, y_train_balanced = self.apply_combined_sampling(X_train, y_train, **sampling_kwargs)
        elif sampling_method == 'oversample':
            X_train_balanced, y_train_balanced = self.simple_random_oversample(X_train, y_train, **sampling_kwargs)
        else:
            logger.warning(f"Unknown sampling method: {sampling_method}. Using original training data.")
            X_train_balanced, y_train_balanced = X_train, y_train
            
        logger.info(f"Balanced train set shape: {X_train_balanced.shape}")
        logger.info(f"Balanced train class distribution: {Counter(y_train_balanced)}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test 