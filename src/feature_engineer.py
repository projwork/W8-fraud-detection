"""
Feature engineering module for fraud detection project.
Handles time-based features, transaction frequency, and velocity calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to handle feature engineering tasks."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        pass
    
    def create_time_features(self, df: pd.DataFrame, datetime_col: str = 'purchase_time') -> pd.DataFrame:
        """
        Create time-based features from datetime column.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            pd.DataFrame: DataFrame with time-based features
        """
        df_processed = df.copy()
        
        if datetime_col not in df_processed.columns:
            logger.warning(f"Column {datetime_col} not found in dataset")
            return df_processed
            
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df_processed[datetime_col]):
            df_processed[datetime_col] = pd.to_datetime(df_processed[datetime_col])
            
        # Extract time components
        df_processed['hour_of_day'] = df_processed[datetime_col].dt.hour
        df_processed['day_of_week'] = df_processed[datetime_col].dt.dayofweek
        df_processed['day_of_month'] = df_processed[datetime_col].dt.day
        df_processed['month'] = df_processed[datetime_col].dt.month
        df_processed['year'] = df_processed[datetime_col].dt.year
        df_processed['quarter'] = df_processed[datetime_col].dt.quarter
        
        # Create categorical time features
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
        df_processed['is_night'] = ((df_processed['hour_of_day'] >= 22) | 
                                   (df_processed['hour_of_day'] <= 6)).astype(int)
        df_processed['is_business_hours'] = ((df_processed['hour_of_day'] >= 9) & 
                                           (df_processed['hour_of_day'] <= 17) & 
                                           (df_processed['day_of_week'] <= 4)).astype(int)
        
        # Time period categories
        def categorize_hour(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
                
        df_processed['time_period'] = df_processed['hour_of_day'].apply(categorize_hour)
        
        logger.info(f"Created time-based features from {datetime_col}")
        return df_processed
    
    def calculate_time_since_signup(self, df: pd.DataFrame, 
                                  signup_col: str = 'signup_time',
                                  purchase_col: str = 'purchase_time') -> pd.DataFrame:
        """
        Calculate time difference between signup and purchase.
        
        Args:
            df: Input DataFrame
            signup_col: Name of signup time column
            purchase_col: Name of purchase time column
            
        Returns:
            pd.DataFrame: DataFrame with time since signup features
        """
        df_processed = df.copy()
        
        if signup_col not in df_processed.columns or purchase_col not in df_processed.columns:
            logger.warning(f"Required columns {signup_col} or {purchase_col} not found")
            return df_processed
            
        # Ensure datetime format
        for col in [signup_col, purchase_col]:
            if not pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                df_processed[col] = pd.to_datetime(df_processed[col])
        
        # Calculate time differences
        time_diff = df_processed[purchase_col] - df_processed[signup_col]
        
        df_processed['time_since_signup_seconds'] = time_diff.dt.total_seconds()
        df_processed['time_since_signup_minutes'] = df_processed['time_since_signup_seconds'] / 60
        df_processed['time_since_signup_hours'] = df_processed['time_since_signup_minutes'] / 60
        df_processed['time_since_signup_days'] = df_processed['time_since_signup_hours'] / 24
        
        # Create categorical features
        df_processed['signup_to_purchase_category'] = pd.cut(
            df_processed['time_since_signup_hours'],
            bins=[-np.inf, 1, 24, 168, 720, np.inf],  # 1hr, 1day, 1week, 1month
            labels=['Immediate', 'Same_day', 'Same_week', 'Same_month', 'Long_term']
        )
        
        # Flag suspiciously quick purchases (potential fraud indicator)
        df_processed['very_quick_purchase'] = (df_processed['time_since_signup_minutes'] < 5).astype(int)
        df_processed['instant_purchase'] = (df_processed['time_since_signup_seconds'] < 60).astype(int)
        
        logger.info("Calculated time since signup features")
        return df_processed
    
    def calculate_transaction_frequency(self, df: pd.DataFrame, 
                                      user_col: str = 'user_id',
                                      device_col: str = 'device_id',
                                      time_col: str = 'purchase_time') -> pd.DataFrame:
        """
        Calculate transaction frequency and velocity features.
        
        Args:
            df: Input DataFrame
            user_col: Name of user ID column
            device_col: Name of device ID column
            time_col: Name of time column
            
        Returns:
            pd.DataFrame: DataFrame with frequency features
        """
        df_processed = df.copy()
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df_processed[time_col]):
            df_processed[time_col] = pd.to_datetime(df_processed[time_col])
        
        # User-based frequency features
        if user_col in df_processed.columns:
            agg_dict = {time_col: ['count', 'min', 'max']}
            if 'purchase_value' in df_processed.columns:
                agg_dict['purchase_value'] = ['sum', 'mean', 'std']
            
            user_stats = df_processed.groupby(user_col).agg(agg_dict).round(2)
            
            # Flatten column names
            user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
            user_stats = user_stats.rename(columns={
                f'{time_col}_count': 'user_transaction_count',
                f'{time_col}_min': 'user_first_transaction',
                f'{time_col}_max': 'user_last_transaction'
            })
            
            if 'purchase_value' in df_processed.columns:
                user_stats = user_stats.rename(columns={
                    'purchase_value_sum': 'user_total_spent',
                    'purchase_value_mean': 'user_avg_transaction_value',
                    'purchase_value_std': 'user_transaction_value_std'
                })
                user_stats['user_transaction_value_std'] = user_stats['user_transaction_value_std'].fillna(0)
            
            # Calculate user activity duration
            user_stats['user_activity_duration_days'] = (
                user_stats['user_last_transaction'] - user_stats['user_first_transaction']
            ).dt.days
            
            # Calculate transaction velocity (transactions per day)
            user_stats['user_transaction_velocity'] = (
                user_stats['user_transaction_count'] / 
                (user_stats['user_activity_duration_days'] + 1)  # Add 1 to avoid division by zero
            )
            
            # Merge back to main dataframe
            df_processed = df_processed.merge(user_stats.reset_index(), on=user_col, how='left')
        
        # Device-based frequency features
        if device_col in df_processed.columns:
            device_stats = df_processed.groupby(device_col).agg({
                time_col: 'count',
                user_col: 'nunique'
            }).round(2)
            
            device_stats.columns = ['device_transaction_count', 'device_unique_users']
            
            # Flag shared devices (potential fraud indicator)
            device_stats['device_shared'] = (device_stats['device_unique_users'] > 1).astype(int)
            device_stats['device_high_activity'] = (device_stats['device_transaction_count'] > 
                                                   device_stats['device_transaction_count'].quantile(0.95)).astype(int)
            
            # Merge back to main dataframe
            df_processed = df_processed.merge(device_stats.reset_index(), on=device_col, how='left')
        
        # Time-window based features (last 24 hours, 7 days, etc.)
        df_processed = self._calculate_time_window_features(df_processed, user_col, time_col)
        
        logger.info("Calculated transaction frequency and velocity features")
        return df_processed
    
    def _calculate_time_window_features(self, df: pd.DataFrame, 
                                       user_col: str, 
                                       time_col: str) -> pd.DataFrame:
        """
        Calculate features within specific time windows.
        
        Args:
            df: Input DataFrame
            user_col: Name of user ID column
            time_col: Name of time column
            
        Returns:
            pd.DataFrame: DataFrame with time window features
        """
        df_processed = df.copy()
        df_processed = df_processed.sort_values([user_col, time_col])
        
        # Initialize new columns
        for window in ['1h', '24h', '7d']:
            df_processed[f'user_transactions_last_{window}'] = 0
            if 'purchase_value' in df_processed.columns:
                df_processed[f'user_spent_last_{window}'] = 0.0
        
        # Calculate for each user
        for user_id in df_processed[user_col].unique():
            user_mask = df_processed[user_col] == user_id
            user_data = df_processed[user_mask].copy()
            
            for idx, row in user_data.iterrows():
                current_time = row[time_col]
                
                # Define time windows
                time_windows = {
                    '1h': current_time - timedelta(hours=1),
                    '24h': current_time - timedelta(hours=24),
                    '7d': current_time - timedelta(days=7)
                }
                
                for window_name, start_time in time_windows.items():
                    # Count transactions in window (excluding current transaction)
                    window_mask = (user_data[time_col] >= start_time) & (user_data[time_col] < current_time)
                    transaction_count = window_mask.sum()
                    
                    df_processed.loc[idx, f'user_transactions_last_{window_name}'] = transaction_count
                    
                    if 'purchase_value' in df_processed.columns:
                        spent_amount = user_data[window_mask]['purchase_value'].sum()
                        df_processed.loc[idx, f'user_spent_last_{window_name}'] = spent_amount
        
        return df_processed
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that might indicate anomalous behavior.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with anomaly detection features
        """
        df_processed = df.copy()
        
        # Purchase value anomalies
        if 'purchase_value' in df_processed.columns:
            # Z-score based anomalies
            mean_value = df_processed['purchase_value'].mean()
            std_value = df_processed['purchase_value'].std()
            df_processed['purchase_value_zscore'] = (df_processed['purchase_value'] - mean_value) / std_value
            df_processed['high_value_transaction'] = (abs(df_processed['purchase_value_zscore']) > 3).astype(int)
            
            # Percentile based anomalies
            p95 = df_processed['purchase_value'].quantile(0.95)
            p99 = df_processed['purchase_value'].quantile(0.99)
            df_processed['top_5_percent_value'] = (df_processed['purchase_value'] > p95).astype(int)
            df_processed['top_1_percent_value'] = (df_processed['purchase_value'] > p99).astype(int)
        
        # Age anomalies
        if 'age' in df_processed.columns:
            # Flag unusual ages
            df_processed['unusual_age'] = ((df_processed['age'] < 13) | (df_processed['age'] > 100)).astype(int)
            df_processed['minor_transaction'] = (df_processed['age'] < 18).astype(int)
        
        # Time-based anomalies
        if 'hour_of_day' in df_processed.columns:
            # Unusual hour transactions
            df_processed['unusual_hour'] = df_processed['hour_of_day'].isin([2, 3, 4, 5]).astype(int)
        
        # Velocity anomalies
        if 'user_transaction_velocity' in df_processed.columns:
            velocity_p95 = df_processed['user_transaction_velocity'].quantile(0.95)
            df_processed['high_velocity_user'] = (df_processed['user_transaction_velocity'] > velocity_p95).astype(int)
        
        logger.info("Created anomaly detection features")
        return df_processed
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        df_processed = df.copy()
        
        # Purchase value and time interactions
        if all(col in df_processed.columns for col in ['purchase_value', 'hour_of_day']):
            df_processed['value_hour_interaction'] = df_processed['purchase_value'] * df_processed['hour_of_day']
        
        if all(col in df_processed.columns for col in ['purchase_value', 'is_weekend']):
            df_processed['value_weekend_interaction'] = df_processed['purchase_value'] * df_processed['is_weekend']
        
        # Age and transaction patterns
        if all(col in df_processed.columns for col in ['age', 'user_transaction_count']):
            df_processed['age_frequency_interaction'] = df_processed['age'] * df_processed['user_transaction_count']
        
        # Time since signup and purchase value
        if all(col in df_processed.columns for col in ['time_since_signup_hours', 'purchase_value']):
            df_processed['signup_value_interaction'] = (df_processed['time_since_signup_hours'] * 
                                                       df_processed['purchase_value'])
        
        logger.info("Created interaction features")
        return df_processed 