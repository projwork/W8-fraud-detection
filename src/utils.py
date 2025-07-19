"""
Utility functions for fraud detection project.
Contains helper functions and common utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=getattr(logging, log_level), format=log_format)

def save_processed_data(df: pd.DataFrame, filepath: str, format: str = 'csv') -> None:
    """
    Save processed DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('csv', 'parquet', 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    logger.info(f"Saved processed data to {filepath}")

def load_processed_data(filepath: str, format: str = 'csv') -> pd.DataFrame:
    """
    Load processed DataFrame from file.
    
    Args:
        filepath: Input file path
        format: File format ('csv', 'parquet', 'pickle')
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'csv':
        df = pd.read_csv(filepath)
    elif format == 'parquet':
        df = pd.read_parquet(filepath)
    elif format == 'pickle':
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    logger.info(f"Loaded processed data from {filepath}")
    return df

def save_model_artifacts(artifacts: Dict[str, Any], output_dir: str) -> None:
    """
    Save model artifacts (scalers, encoders, models, etc.).
    
    Args:
        artifacts: Dictionary of artifacts to save
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, artifact in artifacts.items():
        filepath = output_dir / f"{name}.joblib"
        joblib.dump(artifact, filepath)
        logger.info(f"Saved {name} to {filepath}")

def load_model_artifacts(artifact_dir: str) -> Dict[str, Any]:
    """
    Load model artifacts from directory.
    
    Args:
        artifact_dir: Directory containing artifacts
        
    Returns:
        Dict: Dictionary of loaded artifacts
    """
    artifact_dir = Path(artifact_dir)
    artifacts = {}
    
    for filepath in artifact_dir.glob("*.joblib"):
        name = filepath.stem
        artifacts[name] = joblib.load(filepath)
        logger.info(f"Loaded {name} from {filepath}")
        
    return artifacts

def create_feature_summary(df: pd.DataFrame, target_col: str = 'class') -> pd.DataFrame:
    """
    Create a summary of features in the dataset.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        pd.DataFrame: Feature summary
    """
    summary_data = []
    
    for column in df.columns:
        if column == target_col:
            continue
            
        feature_info = {
            'feature': column,
            'dtype': str(df[column].dtype),
            'null_count': df[column].isnull().sum(),
            'null_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'unique_values': df[column].nunique(),
            'is_numeric': pd.api.types.is_numeric_dtype(df[column]),
            'is_categorical': pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])
        }
        
        if feature_info['is_numeric']:
            feature_info.update({
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'median': df[column].median()
            })
        
        summary_data.append(feature_info)
    
    return pd.DataFrame(summary_data)

def plot_feature_importance(importance_scores: Dict[str, float], 
                          title: str = "Feature Importance",
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance scores.
    
    Args:
        importance_scores: Dictionary of feature names and importance scores
        title: Plot title
        top_n: Number of top features to display
        figsize: Figure size
    """
    # Sort features by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    features, scores = zip(*sorted_features)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def calculate_fraud_rate_by_feature(df: pd.DataFrame, 
                                  feature_col: str, 
                                  target_col: str = 'class') -> pd.DataFrame:
    """
    Calculate fraud rate by feature values.
    
    Args:
        df: Input DataFrame
        feature_col: Feature column to analyze
        target_col: Target column name
        
    Returns:
        pd.DataFrame: Fraud rate analysis
    """
    if target_col not in df.columns and 'Class' in df.columns:
        target_col = 'Class'
        
    fraud_rates = df.groupby(feature_col)[target_col].agg(['count', 'sum'])
    fraud_rates['fraud_rate'] = fraud_rates['sum'] / fraud_rates['count']
    fraud_rates['normal_count'] = fraud_rates['count'] - fraud_rates['sum']
    fraud_rates = fraud_rates.rename(columns={'sum': 'fraud_count', 'count': 'total_count'})
    
    return fraud_rates.round(4)

def compare_distributions(df1: pd.DataFrame, df2: pd.DataFrame, 
                        columns: List[str], 
                        labels: List[str] = ['Dataset 1', 'Dataset 2'],
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Compare distributions of features between two datasets.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        columns: List of columns to compare
        labels: Labels for the datasets
        figsize: Figure size
    """
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    
    for i, column in enumerate(columns):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col] if n_cols > 1 else axes[row]
        
        if column in df1.columns and column in df2.columns:
            if pd.api.types.is_numeric_dtype(df1[column]):
                # Histogram for numeric columns
                df1[column].hist(alpha=0.7, label=labels[0], bins=30, ax=ax, density=True)
                df2[column].hist(alpha=0.7, label=labels[1], bins=30, ax=ax, density=True)
                ax.set_xlabel(column)
                ax.set_ylabel('Density')
                ax.legend()
            else:
                # Bar plot for categorical columns
                combined_values = list(set(df1[column].unique()) | set(df2[column].unique()))[:10]
                df1_counts = df1[column].value_counts()
                df2_counts = df2[column].value_counts()
                
                x = range(len(combined_values))
                width = 0.35
                
                ax.bar([i - width/2 for i in x], 
                      [df1_counts.get(val, 0) for val in combined_values],
                      width, label=labels[0], alpha=0.7)
                ax.bar([i + width/2 for i in x], 
                      [df2_counts.get(val, 0) for val in combined_values],
                      width, label=labels[1], alpha=0.7)
                
                ax.set_xlabel(column)
                ax.set_ylabel('Count')
                ax.set_xticks(x)
                ax.set_xticklabels(combined_values, rotation=45)
                ax.legend()
        
        ax.set_title(f'Distribution Comparison - {column}')
    
    # Remove empty subplots
    for i in range(len(columns), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row][col] if n_cols > 1 else axes[row])
    
    plt.tight_layout()
    plt.show()

def memory_usage_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of DataFrame by converting to optimal dtypes.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    df_optimized = df.copy()
    
    for column in df_optimized.columns:
        if pd.api.types.is_integer_dtype(df_optimized[column]):
            col_min = df_optimized[column].min()
            col_max = df_optimized[column].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df_optimized[column] = df_optimized[column].astype(np.uint8)
                elif col_max < 65535:
                    df_optimized[column] = df_optimized[column].astype(np.uint16)
                elif col_max < 4294967295:
                    df_optimized[column] = df_optimized[column].astype(np.uint32)
            else:  # Signed integers
                if col_min > -128 and col_max < 127:
                    df_optimized[column] = df_optimized[column].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[column] = df_optimized[column].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[column] = df_optimized[column].astype(np.int32)
        
        elif pd.api.types.is_float_dtype(df_optimized[column]):
            df_optimized[column] = pd.to_numeric(df_optimized[column], downcast='float')
    
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    
    logger.info(f"Memory usage reduced from {original_memory:.2f} MB to {optimized_memory:.2f} MB "
               f"({((original_memory - optimized_memory) / original_memory * 100):.1f}% reduction)")
    
    return df_optimized

def save_experiment_config(config: Dict, output_path: str) -> None:
    """
    Save experiment configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Saved experiment config to {output_path}")

def load_experiment_config(config_path: str) -> Dict:
    """
    Load experiment configuration from JSON file.
    
    Args:
        config_path: Configuration file path
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded experiment config from {config_path}")
    return config 