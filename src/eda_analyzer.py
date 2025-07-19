"""
Exploratory Data Analysis module for fraud detection project.
Handles univariate and bivariate analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class EDAAnalyzer:
    """Class to handle exploratory data analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize EDAAnalyzer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        
    def dataset_overview(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Provide a comprehensive overview of the dataset.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dict: Overview statistics
        """
        overview = {
            'dataset_name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Class distribution if 'class' column exists
        if 'class' in df.columns:
            overview['class_distribution'] = df['class'].value_counts().to_dict()
            overview['class_balance_ratio'] = df['class'].value_counts().min() / df['class'].value_counts().max()
        elif 'Class' in df.columns:
            overview['class_distribution'] = df['Class'].value_counts().to_dict()
            overview['class_balance_ratio'] = df['Class'].value_counts().min() / df['Class'].value_counts().max()
            
        logger.info(f"Generated overview for {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        return overview
    
    def univariate_analysis_numerical(self, df: pd.DataFrame, columns: List[str] = None) -> None:
        """
        Perform univariate analysis for numerical columns.
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns to analyze (if None, auto-detect)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        # Remove ID columns and binary target variables
        columns = [col for col in columns if not col.lower().endswith('_id') and 
                  not col.lower() in ['class', 'user_id', 'device_id']]
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        
        for i, column in enumerate(columns):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_cols > 1 else axes[row]
            
            # Create histogram with KDE
            df[column].hist(bins=50, alpha=0.7, ax=ax, density=True)
            df[column].plot.kde(ax=ax, secondary_y=False)
            ax.set_title(f'Distribution of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            
        # Remove empty subplots
        for i in range(len(columns), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row][col] if n_cols > 1 else axes[row])
            
        plt.tight_layout()
        plt.show()
        
        # Print statistical summary
        print("\nNumerical Features Statistical Summary:")
        print("="*50)
        print(df[columns].describe())
    
    def univariate_analysis_categorical(self, df: pd.DataFrame, columns: List[str] = None) -> None:
        """
        Perform univariate analysis for categorical columns.
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to analyze (if None, auto-detect)
        """
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        for column in columns:
            if df[column].nunique() <= 20:  # Only plot if reasonable number of categories
                plt.figure(figsize=self.figsize)
                
                # Count plot
                plt.subplot(1, 2, 1)
                value_counts = df[column].value_counts()
                sns.countplot(data=df, x=column, order=value_counts.index)
                plt.title(f'Count Plot - {column}')
                plt.xticks(rotation=45)
                
                # Pie chart
                plt.subplot(1, 2, 2)
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                plt.title(f'Pie Chart - {column}')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n{column} - Value Counts:")
                print(value_counts)
                print(f"Unique values: {df[column].nunique()}")
                print("-" * 30)
    
    def bivariate_analysis_with_target(self, df: pd.DataFrame, target_col: str = 'class') -> None:
        """
        Perform bivariate analysis between features and target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
        """
        # Handle different target column names
        if target_col not in df.columns and 'Class' in df.columns:
            target_col = 'Class'
        elif target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in dataset")
            return
            
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature lists
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
        # Numerical features vs target
        print(f"Bivariate Analysis - Numerical Features vs {target_col}")
        print("="*60)
        
        for column in numerical_cols[:6]:  # Limit to first 6 for readability
            plt.figure(figsize=self.figsize)
            
            # Box plot
            plt.subplot(1, 2, 1)
            sns.boxplot(data=df, x=target_col, y=column)
            plt.title(f'{column} by {target_col}')
            
            # Violin plot
            plt.subplot(1, 2, 2)
            sns.violinplot(data=df, x=target_col, y=column)
            plt.title(f'{column} Distribution by {target_col}')
            
            plt.tight_layout()
            plt.show()
            
            # Statistical comparison
            fraud_mean = df[df[target_col] == 1][column].mean()
            normal_mean = df[df[target_col] == 0][column].mean()
            print(f"\n{column}:")
            print(f"  Fraud transactions mean: {fraud_mean:.2f}")
            print(f"  Normal transactions mean: {normal_mean:.2f}")
            print(f"  Difference: {abs(fraud_mean - normal_mean):.2f}")
            
        # Categorical features vs target
        print(f"\nBivariate Analysis - Categorical Features vs {target_col}")
        print("="*60)
        
        for column in categorical_cols[:4]:  # Limit to first 4
            if df[column].nunique() <= 10:  # Only analyze if reasonable number of categories
                plt.figure(figsize=self.figsize)
                
                # Stacked bar plot
                plt.subplot(1, 2, 1)
                fraud_rates = df.groupby(column)[target_col].agg(['count', 'sum'])
                fraud_rates['fraud_rate'] = fraud_rates['sum'] / fraud_rates['count']
                fraud_rates['fraud_rate'].plot(kind='bar')
                plt.title(f'Fraud Rate by {column}')
                plt.ylabel('Fraud Rate')
                plt.xticks(rotation=45)
                
                # Count plot by target
                plt.subplot(1, 2, 2)
                sns.countplot(data=df, x=column, hue=target_col)
                plt.title(f'{column} Count by {target_col}')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n{column} - Fraud Rate Analysis:")
                print(fraud_rates[['count', 'fraud_rate']])
    
    def correlation_analysis(self, df: pd.DataFrame, target_col: str = 'class') -> None:
        """
        Perform correlation analysis.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
        """
        # Handle different target column names
        if target_col not in df.columns and 'Class' in df.columns:
            target_col = 'Class'
            
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) < 2:
            logger.warning("Not enough numerical columns for correlation analysis")
            return
            
        # Correlation matrix
        plt.figure(figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        correlation_matrix = numerical_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()
        
        # Top correlations with target
        if target_col in correlation_matrix.columns:
            target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
            print(f"\nTop correlations with {target_col}:")
            print("="*40)
            for feature, corr in target_corr.head(10).items():
                if feature != target_col:
                    print(f"{feature}: {corr:.3f}")
    
    def class_imbalance_analysis(self, df: pd.DataFrame, target_col: str = 'class') -> Dict:
        """
        Analyze class imbalance in the target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Dict: Class imbalance metrics
        """
        # Handle different target column names
        if target_col not in df.columns and 'Class' in df.columns:
            target_col = 'Class'
        elif target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return {}
            
        value_counts = df[target_col].value_counts()
        
        plt.figure(figsize=self.figsize)
        
        # Bar plot
        plt.subplot(1, 2, 1)
        value_counts.plot(kind='bar')
        plt.title(f'Class Distribution - {target_col}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(value_counts.values, labels=[f'Class {i}' for i in value_counts.index], 
                autopct='%1.2f%%')
        plt.title(f'Class Distribution - {target_col}')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate imbalance metrics
        majority_class = value_counts.max()
        minority_class = value_counts.min()
        imbalance_ratio = minority_class / majority_class
        
        imbalance_metrics = {
            'majority_class_count': majority_class,
            'minority_class_count': minority_class,
            'imbalance_ratio': imbalance_ratio,
            'class_distribution': value_counts.to_dict(),
            'imbalance_severity': 'Severe' if imbalance_ratio < 0.1 else 'Moderate' if imbalance_ratio < 0.3 else 'Mild'
        }
        
        print(f"\nClass Imbalance Analysis for {target_col}:")
        print("="*40)
        print(f"Majority class count: {majority_class:,}")
        print(f"Minority class count: {minority_class:,}")
        print(f"Imbalance ratio: {imbalance_ratio:.4f}")
        print(f"Imbalance severity: {imbalance_metrics['imbalance_severity']}")
        
        return imbalance_metrics 