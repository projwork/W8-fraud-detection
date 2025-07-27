"""
Data splitting module for fraud detection project.
Handles data preparation and train-test splits for model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Union
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataSplitter:
    """Class to handle data preparation and train-test splits."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataSplitter.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.feature_names = {}
        
    def prepare_fraud_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare fraud detection dataset for modeling.
        
        Args:
            df: Input DataFrame with fraud data
            
        Returns:
            Tuple: (features, target)
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Identify target column
        target_col = 'class' if 'class' in data.columns else 'Class'
        if target_col not in data.columns:
            raise ValueError("Target column 'class' or 'Class' not found in dataset")
        
        # Separate features and target
        y = data[target_col]
        
        # Remove non-feature columns
        exclude_cols = [
            target_col, 'user_id', 'device_id', 'signup_time', 
            'purchase_time', 'ip_address', 'ip_address_int'
        ]
        
        # Keep only numerical columns for modeling
        X = data.drop(columns=[col for col in exclude_cols if col in data.columns])
        X = X.select_dtypes(include=[np.number])
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        logger.info(f"Prepared fraud data: {X.shape[0]} samples, {X.shape[1]} features")
        self.feature_names['fraud'] = list(X.columns)
        
        return X, y
    
    def prepare_creditcard_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare credit card dataset for modeling.
        
        Args:
            df: Input DataFrame with credit card data
            
        Returns:
            Tuple: (features, target)
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Identify target column
        target_col = 'Class' if 'Class' in data.columns else 'class'
        if target_col not in data.columns:
            raise ValueError("Target column 'Class' or 'class' not found in dataset")
        
        # Separate features and target
        y = data[target_col]
        X = data.drop(columns=[target_col])
        
        # Ensure all features are numerical
        X = X.select_dtypes(include=[np.number])
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        logger.info(f"Prepared credit card data: {X.shape[0]} samples, {X.shape[1]} features")
        self.feature_names['creditcard'] = list(X.columns)
        
        return X, y
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.2,
                               stratify: bool = True,
                               scale_features: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train-test split with optional scaling.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of test set
            stratify: Whether to stratify the split
            scale_features: Whether to scale features
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Perform train-test split
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Store scaler for future use
            dataset_name = 'fraud' if 'purchase_value' in X.columns else 'creditcard'
            self.scalers[dataset_name] = scaler
            
            X_train, X_test = X_train_scaled, X_test_scaled
            logger.info("Features scaled using StandardScaler")
        
        logger.info(f"Train-test split completed:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        logger.info(f"  Training class distribution: {dict(y_train.value_counts())}")
        logger.info(f"  Test class distribution: {dict(y_test.value_counts())}")
        
        return X_train, X_test, y_train, y_test
    
    def get_cv_folds(self, X: pd.DataFrame, y: pd.Series, 
                     n_splits: int = 5) -> StratifiedKFold:
        """
        Get stratified K-fold cross-validation iterator.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of CV folds
            
        Returns:
            StratifiedKFold: CV iterator
        """
        skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        logger.info(f"Created {n_splits}-fold stratified cross-validation")
        return skf
    
    def prepare_datasets_for_modeling(self, fraud_df: pd.DataFrame, 
                                    creditcard_df: pd.DataFrame,
                                    test_size: float = 0.2) -> Dict:
        """
        Prepare both datasets for modeling.
        
        Args:
            fraud_df: Fraud detection dataset
            creditcard_df: Credit card dataset
            test_size: Test set proportion
            
        Returns:
            Dict: Prepared datasets with train-test splits
        """
        results = {}
        
        # Prepare fraud dataset
        logger.info("Preparing fraud detection dataset...")
        X_fraud, y_fraud = self.prepare_fraud_data(fraud_df)
        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = self.create_train_test_split(
            X_fraud, y_fraud, test_size=test_size
        )
        
        results['fraud'] = {
            'X_train': X_train_fraud,
            'X_test': X_test_fraud,
            'y_train': y_train_fraud,
            'y_test': y_test_fraud,
            'feature_names': self.feature_names['fraud']
        }
        
        # Prepare credit card dataset
        logger.info("Preparing credit card dataset...")
        X_cc, y_cc = self.prepare_creditcard_data(creditcard_df)
        X_train_cc, X_test_cc, y_train_cc, y_test_cc = self.create_train_test_split(
            X_cc, y_cc, test_size=test_size
        )
        
        results['creditcard'] = {
            'X_train': X_train_cc,
            'X_test': X_test_cc,
            'y_train': y_train_cc,
            'y_test': y_test_cc,
            'feature_names': self.feature_names['creditcard']
        }
        
        logger.info("Both datasets prepared for modeling")
        return results
    
    def get_dataset_info(self, datasets: Dict) -> Dict:
        """
        Get information about prepared datasets.
        
        Args:
            datasets: Prepared datasets dictionary
            
        Returns:
            Dict: Dataset information
        """
        info = {}
        
        for dataset_name, data in datasets.items():
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            
            # Calculate class imbalance
            train_counts = y_train.value_counts()
            test_counts = y_test.value_counts()
            
            info[dataset_name] = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1],
                'train_class_distribution': dict(train_counts),
                'test_class_distribution': dict(test_counts),
                'train_imbalance_ratio': train_counts.min() / train_counts.max(),
                'test_imbalance_ratio': test_counts.min() / test_counts.max()
            }
        
        return info 