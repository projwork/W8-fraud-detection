"""
Data loading module for fraud detection project.
Handles loading and initial validation of datasets.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle loading and initial validation of fraud detection datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_fraud_data(self) -> pd.DataFrame:
        """
        Load the main fraud detection dataset.
        
        Returns:
            pd.DataFrame: Loaded fraud data
        """
        file_path = self.data_dir / "Fraud_Data.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Fraud_Data.csv not found in {self.data_dir}")
            
        logger.info(f"Loading fraud data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert datetime columns
        datetime_cols = ['signup_time', 'purchase_time']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        self.datasets['fraud_data'] = df
        logger.info(f"Loaded fraud data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_ip_country_data(self) -> pd.DataFrame:
        """
        Load the IP address to country mapping dataset.
        
        Returns:
            pd.DataFrame: IP to country mapping data
        """
        file_path = self.data_dir / "IpAddress_to_Country.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"IpAddress_to_Country.csv not found in {self.data_dir}")
            
        logger.info(f"Loading IP country data from {file_path}")
        df = pd.read_csv(file_path)
        self.datasets['ip_country'] = df
        logger.info(f"Loaded IP country data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_creditcard_data(self) -> pd.DataFrame:
        """
        Load the credit card fraud dataset.
        
        Returns:
            pd.DataFrame: Credit card fraud data
        """
        file_path = self.data_dir / "creditcard.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"creditcard.csv not found in {self.data_dir}")
            
        logger.info(f"Loading credit card data from {file_path}")
        df = pd.read_csv(file_path)
        self.datasets['creditcard'] = df
        logger.info(f"Loaded credit card data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets at once.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all loaded datasets
        """
        datasets = {
            'fraud_data': self.load_fraud_data(),
            'ip_country': self.load_ip_country_data(),
            'creditcard': self.load_creditcard_data()
        }
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get basic information about a loaded dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict: Basic information about the dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
            
        df = self.datasets[dataset_name]
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
    
    def validate_datasets(self) -> Dict[str, Dict]:
        """
        Validate all loaded datasets and return validation results.
        
        Returns:
            Dict: Validation results for each dataset
        """
        validation_results = {}
        
        for name, df in self.datasets.items():
            validation_results[name] = {
                'has_duplicates': df.duplicated().any(),
                'duplicate_count': df.duplicated().sum(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'data_types_valid': True  # Will be expanded based on specific validations
            }
            
        return validation_results 