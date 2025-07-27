#!/usr/bin/env python3
"""
Script to create the Task 2 notebook for model building and training.
"""

import json
import os

def create_notebook():
    """Create the Task 2 notebook with proper structure."""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Task 2: Model Building and Training\n",
                    "## Fraud Detection Model Development\n",
                    "\n",
                    "This notebook implements Task 2 of the fraud detection project, focusing on:\n",
                    "\n",
                    "1. **Data Preparation**: Separate features and target, train-test split\n",
                    "2. **Model Selection**: Logistic Regression (baseline) + Ensemble Model (Random Forest/LightGBM)\n",
                    "3. **Model Training**: Train models on both datasets with cross-validation\n",
                    "4. **Model Evaluation**: Use appropriate metrics for imbalanced data (AUC-PR, F1-Score, Confusion Matrix)\n",
                    "5. **Model Comparison**: Clear justification of the \"best\" model\n",
                    "\n",
                    "## Datasets:\n",
                    "- **Fraud_Data.csv**: E-commerce transactions (target: 'class')\n",
                    "- **creditcard.csv**: Bank transactions (target: 'Class')\n",
                    "\n",
                    "## Approach:\n",
                    "- Modular programming with custom modules in `/src`\n",
                    "- Comprehensive evaluation with imbalanced-data-appropriate metrics\n",
                    "- Cross-validation for robust performance assessment"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Setup and Imports"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Standard imports\n",
                    "import sys\n",
                    "import os\n",
                    "import warnings\n",
                    "import time\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Add src directory to path for modular imports\n",
                    "sys.path.append('../src')\n",
                    "\n",
                    "# Data manipulation and analysis\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "\n",
                    "# Visualization\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "plt.style.use('seaborn-v0_8')\n",
                    "\n",
                    "# Machine learning basics\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.metrics import classification_report\n",
                    "\n",
                    "# Custom modules for Task 2\n",
                    "from data_splitter import DataSplitter\n",
                    "from model_builder import ModelBuilder\n",
                    "from model_trainer import ModelTrainer\n",
                    "from model_evaluator import ModelEvaluator\n",
                    "\n",
                    "# Data loading (reuse from Task 1)\n",
                    "from data_loader import DataLoader\n",
                    "from utils import setup_logging\n",
                    "\n",
                    "# Set up logging\n",
                    "setup_logging('INFO')\n",
                    "\n",
                    "# Set random seed for reproducibility\n",
                    "np.random.seed(42)\n",
                    "\n",
                    "print(\"✅ All modules imported successfully!\")\n",
                    "print(\"📁 Working directory:\", os.getcwd())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Data Loading"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize data loader\n",
                    "data_loader = DataLoader(data_dir='../data')\n",
                    "\n",
                    "print(\"🔄 Loading datasets for modeling...\")\n",
                    "\n",
                    "# Load datasets\n",
                    "try:\n",
                    "    fraud_data = data_loader.load_fraud_data()\n",
                    "    creditcard_data = data_loader.load_creditcard_data()\n",
                    "    \n",
                    "    print(f\"✅ Fraud data loaded: {fraud_data.shape}\")\n",
                    "    print(f\"✅ Credit card data loaded: {creditcard_data.shape}\")\n",
                    "    \n",
                    "    # Display basic info\n",
                    "    print(f\"\\n📊 Dataset Overview:\")\n",
                    "    print(f\"Fraud data target column: {'class' if 'class' in fraud_data.columns else 'Class'}\")\n",
                    "    print(f\"Credit card target column: {'Class' if 'Class' in creditcard_data.columns else 'class'}\")\n",
                    "    \n",
                    "    # Check class distribution\n",
                    "    fraud_target = 'class' if 'class' in fraud_data.columns else 'Class'\n",
                    "    cc_target = 'Class' if 'Class' in creditcard_data.columns else 'class'\n",
                    "    \n",
                    "    print(f\"\\n🎯 Class Distributions:\")\n",
                    "    print(f\"Fraud data - {fraud_target}:\")\n",
                    "    print(fraud_data[fraud_target].value_counts())\n",
                    "    print(f\"\\nCredit card data - {cc_target}:\")\n",
                    "    print(creditcard_data[cc_target].value_counts())\n",
                    "    \n",
                    "except Exception as e:\n",
                    "    print(f\"❌ Error loading data: {e}\")\n",
                    "    print(\"Make sure the data files are in the '../data' directory\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3. Data Preparation and Train-Test Split"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize data splitter\n",
                    "data_splitter = DataSplitter(random_state=42)\n",
                    "\n",
                    "print(\"🔧 Preparing datasets for modeling...\")\n",
                    "\n",
                    "# Prepare both datasets\n",
                    "datasets = data_splitter.prepare_datasets_for_modeling(\n",
                    "    fraud_df=fraud_data,\n",
                    "    creditcard_df=creditcard_data,\n",
                    "    test_size=0.2\n",
                    ")\n",
                    "\n",
                    "# Get dataset information\n",
                    "dataset_info = data_splitter.get_dataset_info(datasets)\n",
                    "\n",
                    "print(\"\\n📊 PREPARED DATASETS SUMMARY\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "for dataset_name, info in dataset_info.items():\n",
                    "    print(f\"\\n{dataset_name.upper()} Dataset:\")\n",
                    "    print(f\"  Training samples: {info['train_samples']:,}\")\n",
                    "    print(f\"  Test samples: {info['test_samples']:,}\")\n",
                    "    print(f\"  Features: {info['n_features']}\")\n",
                    "    print(f\"  Train class distribution: {info['train_class_distribution']}\")\n",
                    "    print(f\"  Test class distribution: {info['test_class_distribution']}\")\n",
                    "    print(f\"  Train imbalance ratio: {info['train_imbalance_ratio']:.4f}\")\n",
                    "    print(f\"  Test imbalance ratio: {info['test_imbalance_ratio']:.4f}\")\n",
                    "\n",
                    "print(\"\\n✅ Data preparation completed!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    """Create the Task 2 notebook."""
    os.makedirs('notebooks', exist_ok=True)
    
    notebook = create_notebook()
    
    notebook_path = 'notebooks/model_building_training.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Task 2 notebook created: {notebook_path}")
    print("📝 The notebook includes the basic structure for model building and training")
    print("💡 You can expand it by adding more cells for training, evaluation, and visualization")

if __name__ == "__main__":
    main() 