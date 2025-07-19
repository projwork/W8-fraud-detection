"""
Data preprocessing module for fraud detection project.
Handles missing values, data cleaning, and basic transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import ipaddress

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class to handle data preprocessing tasks."""
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.scalers = {}
        self.encoders = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Dictionary specifying strategy for each column
                     Options: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        if strategy is None:
            # Default strategy
            strategy = {}
            
        # Report missing values
        missing_summary = df_processed.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        
        if len(missing_summary) > 0:
            logger.info(f"Missing values found in columns: {missing_summary.to_dict()}")
            
            for column in missing_summary.index:
                col_strategy = strategy.get(column, 'drop')
                
                if col_strategy == 'drop':
                    df_processed = df_processed.dropna(subset=[column])
                elif col_strategy == 'mean' and df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].mean(), inplace=True)
                elif col_strategy == 'median' and df_processed[column].dtype in ['int64', 'float64']:
                    df_processed[column].fillna(df_processed[column].median(), inplace=True)
                elif col_strategy == 'mode':
                    mode_value = df_processed[column].mode()[0] if not df_processed[column].mode().empty else 'Unknown'
                    df_processed[column].fillna(mode_value, inplace=True)
                elif col_strategy == 'forward_fill':
                    df_processed[column].fillna(method='ffill', inplace=True)
                elif col_strategy == 'backward_fill':
                    df_processed[column].fillna(method='bfill', inplace=True)
                    
        logger.info(f"Missing values handled. Shape: {df_processed.shape}")
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input DataFrame
            subset: Column names to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        initial_shape = df.shape
        df_processed = df.drop_duplicates(subset=subset, keep=keep)
        
        duplicates_removed = initial_shape[0] - df_processed.shape[0]
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            
        return df_processed
    
    def correct_data_types(self, df: pd.DataFrame, type_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Correct data types for columns.
        
        Args:
            df: Input DataFrame
            type_mapping: Dictionary mapping column names to desired data types
        
        Returns:
            pd.DataFrame: DataFrame with corrected data types
        """
        df_processed = df.copy()
        
        if type_mapping:
            for column, dtype in type_mapping.items():
                if column in df_processed.columns:
                    try:
                        if dtype == 'datetime':
                            df_processed[column] = pd.to_datetime(df_processed[column])
                        else:
                            df_processed[column] = df_processed[column].astype(dtype)
                        logger.info(f"Converted {column} to {dtype}")
                    except Exception as e:
                        logger.warning(f"Could not convert {column} to {dtype}: {e}")
                        
        return df_processed
    
    def ip_to_int(self, ip_address: str) -> Optional[int]:
        """
        Convert IP address to integer format.
        
        Args:
            ip_address: IP address as string
            
        Returns:
            int: IP address as integer, None if invalid
        """
        try:
            return int(ipaddress.IPv4Address(ip_address))
        except:
            return None
    
    def convert_ip_addresses(self, df: pd.DataFrame, ip_column: str = 'ip_address') -> pd.DataFrame:
        """
        Convert IP addresses to integer format for easier processing.
        
        Args:
            df: Input DataFrame
            ip_column: Name of the IP address column
            
        Returns:
            pd.DataFrame: DataFrame with IP addresses converted to integers
        """
        df_processed = df.copy()
        
        if ip_column in df_processed.columns:
            df_processed[f'{ip_column}_int'] = df_processed[ip_column].apply(self.ip_to_int)
            invalid_ips = df_processed[f'{ip_column}_int'].isnull().sum()
            if invalid_ips > 0:
                logger.warning(f"Found {invalid_ips} invalid IP addresses")
                
        return df_processed
    
    def merge_ip_geolocation(self, fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge fraud data with IP geolocation data.
        
        Args:
            fraud_df: Main fraud dataset with IP addresses
            ip_country_df: IP to country mapping dataset
            
        Returns:
            pd.DataFrame: Merged dataset with country information
        """
        # Convert IP addresses to integers if not already done
        if 'ip_address_int' not in fraud_df.columns:
            fraud_df = self.convert_ip_addresses(fraud_df)
            
        # Perform the merge using IP ranges
        merged_df = fraud_df.copy()
        merged_df['country'] = None
        
        for idx, row in ip_country_df.iterrows():
            mask = ((merged_df['ip_address_int'] >= row['lower_bound_ip_address']) & 
                   (merged_df['ip_address_int'] <= row['upper_bound_ip_address']))
            merged_df.loc[mask, 'country'] = row['country']
            
        logger.info(f"Merged geolocation data. {merged_df['country'].notna().sum()} IPs matched to countries")
        return merged_df
    
    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str], 
                                  method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            method: Encoding method ('onehot', 'label')
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        df_processed = df.copy()
        
        for column in columns:
            if column in df_processed.columns:
                if method == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(df_processed[column], prefix=column)
                    df_processed = pd.concat([df_processed, dummies], axis=1)
                    df_processed = df_processed.drop(column, axis=1)
                    
                elif method == 'label':
                    # Label encoding
                    if column not in self.encoders:
                        self.encoders[column] = LabelEncoder()
                        df_processed[f'{column}_encoded'] = self.encoders[column].fit_transform(df_processed[column])
                    else:
                        df_processed[f'{column}_encoded'] = self.encoders[column].transform(df_processed[column])
                        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns to scale
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        df_processed = df.copy()
        
        if method not in self.scalers:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'minmax':
                self.scalers[method] = MinMaxScaler()
                
        # Scale only the specified columns
        df_processed[columns] = self.scalers[method].fit_transform(df_processed[columns])
        
        logger.info(f"Scaled {len(columns)} features using {method} scaling")
        return df_processed 