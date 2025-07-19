#!/usr/bin/env python3
"""
Test script for fraud detection project modules.
Validates that all components can be imported and basic functionality works.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test all module imports."""
    print("🧪 Testing Module Imports")
    print("=" * 50)
    
    try:
        from data_loader import DataLoader
        print("✅ DataLoader imported successfully")
        
        from data_preprocessor import DataPreprocessor
        print("✅ DataPreprocessor imported successfully")
        
        from eda_analyzer import EDAAnalyzer
        print("✅ EDAAnalyzer imported successfully")
        
        from feature_engineer import FeatureEngineer
        print("✅ FeatureEngineer imported successfully")
        
        from imbalance_handler import ImbalanceHandler
        print("✅ ImbalanceHandler imported successfully")
        
        from utils import setup_logging, create_feature_summary
        print("✅ Utils imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        # Test DataLoader initialization
        from data_loader import DataLoader
        loader = DataLoader(data_dir='data')
        print("✅ DataLoader initialization successful")
        
        # Test DataPreprocessor
        from data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        print("✅ DataPreprocessor initialization successful")
        
        # Test EDAAnalyzer
        from eda_analyzer import EDAAnalyzer
        eda = EDAAnalyzer()
        print("✅ EDAAnalyzer initialization successful")
        
        # Test FeatureEngineer
        from feature_engineer import FeatureEngineer
        engineer = FeatureEngineer()
        print("✅ FeatureEngineer initialization successful")
        
        # Test ImbalanceHandler
        from imbalance_handler import ImbalanceHandler
        handler = ImbalanceHandler()
        print("✅ ImbalanceHandler initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_dependencies():
    """Test critical dependencies."""
    print("\n📦 Testing Dependencies")
    print("=" * 50)
    
    dependencies = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'imblearn', 'joblib'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"❌ {dep} is missing")
            return False
    
    return True

def test_project_structure():
    """Test project directory structure."""
    print("\n📁 Testing Project Structure")
    print("=" * 50)
    
    required_dirs = ['src', 'data', 'notebooks']
    required_files = [
        'src/data_loader.py',
        'src/data_preprocessor.py', 
        'src/eda_analyzer.py',
        'src/feature_engineer.py',
        'src/imbalance_handler.py',
        'src/utils.py',
        'notebooks/fraud_detection_analysis.ipynb',
        'requirements.txt',
        'README.md'
    ]
    
    # Check directories
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory {directory} exists")
        else:
            print(f"❌ Directory {directory} missing")
            return False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Fraud Detection Project - Module Testing")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The project is ready to use.")
        print("\n📋 Next Steps:")
        print("1. Ensure your data files are in the 'data' directory:")
        print("   - Fraud_Data.csv")
        print("   - IpAddress_to_Country.csv") 
        print("   - creditcard.csv")
        print("2. Start Jupyter: jupyter notebook")
        print("3. Open notebooks/fraud_detection_analysis.ipynb")
        print("4. Run the analysis!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 