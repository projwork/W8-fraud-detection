# Fraud Detection Analysis Project

A comprehensive fraud detection analysis system built with a modular Python approach. This project implements Task 1 of fraud detection analysis, covering data preprocessing, exploratory data analysis (EDA), feature engineering, and class imbalance handling.

## 🏗️ Project Structure

```
fraud-detection/
├── data/                           # Data directory
│   ├── Fraud_Data.csv             # E-commerce transaction data
│   ├── IpAddress_to_Country.csv   # IP geolocation mapping
│   ├── creditcard.csv             # Bank transaction data
│   └── processed/                 # Processed datasets (generated)
├── src/                           # Modular source code
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # Data loading and validation
│   ├── data_preprocessor.py      # Data preprocessing
│   ├── eda_analyzer.py           # Exploratory data analysis
│   ├── feature_engineer.py       # Feature engineering
│   ├── imbalance_handler.py      # Class imbalance handling
│   └── utils.py                  # Utility functions
├── notebooks/                     # Jupyter notebooks
│   └── fraud_detection_analysis.ipynb  # Main analysis notebook
├── scripts/                       # Additional scripts
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── setup_environment.py         # Environment setup script
└── README.md                     # This file
```

## 📊 Datasets

### 1. Fraud_Data.csv

E-commerce transaction data for fraud detection:

- **user_id**: Unique user identifier
- **signup_time**: User registration timestamp
- **purchase_time**: Transaction timestamp
- **purchase_value**: Transaction amount ($)
- **device_id**: Device identifier
- **source**: Traffic source (SEO, Ads, etc.)
- **browser**: Browser used
- **sex**: User gender
- **age**: User age
- **ip_address**: Transaction IP address
- **class**: Target variable (1=fraud, 0=legitimate)

### 2. IpAddress_to_Country.csv

IP geolocation mapping:

- **lower_bound_ip_address**: IP range start
- **upper_bound_ip_address**: IP range end
- **country**: Country name

### 3. creditcard.csv

Bank transaction data:

- **Time**: Seconds from first transaction
- **V1-V28**: PCA-transformed features
- **Amount**: Transaction amount
- **Class**: Target variable (1=fraud, 0=legitimate)

## 🚀 Getting Started

### Prerequisites

- Python 3.12+ (compatible with Python 3.8+)
- Virtual environment

### Installation

1. **Clone or navigate to the project directory**:

   ```bash
   cd fraud-detection
   ```

2. **Create and activate virtual environment**:

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate (Windows)
   .venv\Scripts\activate

   # Activate (Mac/Linux)
   source .venv/bin/activate
   ```

3. **Run the setup script**:

   ```bash
   python setup_environment.py
   ```

4. **Or install manually**:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

1. **Activate environment**:

   ```bash
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

2. **Start Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

3. **Open the main analysis notebook**:
   Navigate to `notebooks/fraud_detection_analysis.ipynb`

## 🔧 Modular Components

### DataLoader (`src/data_loader.py`)

- Load and validate datasets
- Handle different file formats
- Data integrity checks

### DataPreprocessor (`src/data_preprocessor.py`)

- Missing value handling
- Duplicate removal
- IP address conversion
- Categorical encoding
- Feature scaling

### EDAAnalyzer (`src/eda_analyzer.py`)

- Dataset overview and statistics
- Univariate analysis (numerical & categorical)
- Bivariate analysis with target variable
- Correlation analysis
- Class imbalance visualization

### FeatureEngineer (`src/feature_engineer.py`)

- Time-based features (hour, day of week, etc.)
- Time since signup calculations
- Transaction frequency and velocity
- Anomaly detection features
- Interaction features

### ImbalanceHandler (`src/imbalance_handler.py`)

- Class distribution analysis
- SMOTE oversampling
- Random undersampling
- Combined sampling techniques
- Strategy recommendations

### Utils (`src/utils.py`)

- Logging setup
- Data saving/loading
- Feature summaries
- Memory optimization
- Configuration management

## 📈 Analysis Pipeline

The complete analysis follows this pipeline:

1. **Data Loading & Validation**

   - Load all three datasets
   - Validate data integrity
   - Check for missing values and duplicates

2. **Data Cleaning & Preprocessing**

   - Handle missing values using appropriate strategies
   - Remove duplicates
   - Convert IP addresses to integers
   - Merge geolocation data

3. **Exploratory Data Analysis**

   - Dataset overviews and statistics
   - Univariate and bivariate analysis
   - Correlation analysis
   - Class imbalance assessment

4. **Feature Engineering**

   - Time-based features from timestamps
   - User behavior patterns
   - Transaction frequency metrics
   - Anomaly indicators
   - Feature interactions

5. **Data Transformation**

   - One-hot encoding for categorical features
   - Standard scaling for numerical features
   - Final dataset preparation

6. **Class Imbalance Handling**
   - Analyze imbalance severity
   - Apply SMOTE for oversampling
   - Create balanced train-test splits
   - Recommendation system for sampling strategies

## 🎯 Key Features

### ✅ Comprehensive Data Processing

- **Missing Value Handling**: Multiple strategies (median, mode, drop)
- **Data Cleaning**: Duplicate removal and data type correction
- **IP Geolocation**: Merge transaction data with country information

### ✅ Advanced Feature Engineering

- **Time Features**: Hour of day, day of week, business hours
- **User Behavior**: Transaction frequency, velocity, patterns
- **Anomaly Detection**: Unusual values, patterns, timing
- **Interactions**: Feature combinations and ratios

### ✅ Class Imbalance Solutions

- **Multiple Techniques**: SMOTE, undersampling, combined methods
- **Smart Recommendations**: Automatic strategy selection
- **Evaluation**: Comprehensive imbalance analysis

### ✅ Modular Architecture

- **Reusable Components**: Independent, testable modules
- **Clean Code**: Well-documented, maintainable structure
- **Flexible Configuration**: Easy parameter adjustment

## 📊 Results

The notebook generates:

- **Processed Datasets**: Clean, feature-engineered data
- **Visualizations**: Comprehensive EDA plots and charts
- **Feature Summary**: Detailed feature analysis
- **Balanced Datasets**: Ready for machine learning
- **Configuration Files**: Experiment tracking

## 🔍 Class Imbalance Challenge

Both datasets exhibit severe class imbalance:

- **Fraud Data**: Highly imbalanced with few fraud cases
- **Credit Card Data**: Extremely imbalanced (~0.17% fraud)

**Solutions Implemented**:

- SMOTE for oversampling minority class
- Random undersampling for majority class
- Combined techniques (SMOTE + Tomek, SMOTE + ENN)
- Intelligent strategy recommendations

## 🛠️ Technical Requirements

### Python Packages

- **Core**: pandas, numpy, scipy
- **ML**: scikit-learn, imbalanced-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Notebook**: jupyter, ipywidgets
- **Utilities**: joblib, tqdm

### System Requirements

- **Memory**: 8GB+ RAM recommended
- **Storage**: 1GB+ free space
- **Python**: 3.8+ (3.12+ recommended)

## 📁 Output Files

The analysis generates several output files in `data/processed/`:

- `fraud_X_train.csv` / `fraud_X_test.csv`: Processed fraud detection features
- `fraud_y_train.csv` / `fraud_y_test.csv`: Fraud detection targets
- `creditcard_X_train.csv` / `creditcard_X_test.csv`: Credit card features
- `creditcard_y_train.csv` / `creditcard_y_test.csv`: Credit card targets
- `feature_summary.csv`: Comprehensive feature analysis
- `experiment_config.json`: Analysis configuration and metadata

## 🤝 Usage Examples

### Using Individual Modules

```python
# Load data
from src.data_loader import DataLoader
loader = DataLoader(data_dir='data')
datasets = loader.load_all_datasets()

# Preprocess data
from src.data_preprocessor import DataPreprocessor
preprocessor = DataPreprocessor()
clean_data = preprocessor.handle_missing_values(data)

# Feature engineering
from src.feature_engineer import FeatureEngineer
engineer = FeatureEngineer()
engineered_data = engineer.create_time_features(data)

# Handle class imbalance
from src.imbalance_handler import ImbalanceHandler
handler = ImbalanceHandler()
X_train, X_test, y_train, y_test = handler.create_balanced_train_test_split(X, y)
```

## 🔬 Next Steps

This project completes **Task 1** of the fraud detection analysis. Next phases could include:

1. **Model Development**: Train various ML algorithms
2. **Model Evaluation**: Compare performance metrics
3. **Feature Selection**: Identify most important features
4. **Hyperparameter Tuning**: Optimize model parameters
5. **Deployment**: Create production-ready pipeline

## 📝 Notes

- **Performance**: Large datasets may require sampling for visualization
- **Memory**: Monitor memory usage with large feature sets
- **Flexibility**: All parameters are configurable
- **Extensibility**: Easy to add new features or preprocessing steps

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce sample size or use chunking
2. **Import Error**: Ensure all dependencies are installed
3. **File Not Found**: Check data file paths
4. **SMOTE Error**: Fallback to simple oversampling implemented

### Performance Tips

- Use sampling for large datasets during EDA
- Enable parallel processing where available
- Monitor memory usage during feature engineering
- Save intermediate results for iterative development

---

**Happy Fraud Detection! 🚀**
