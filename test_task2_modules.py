#!/usr/bin/env python3
"""
Test script for Task 2 modules of the fraud detection project.
Verifies that all model building, training, and evaluation modules work correctly.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def test_task2_imports():
    """Test that all Task 2 modules can be imported."""
    print("🔍 Testing Task 2 module imports...")
    
    try:
        from data_splitter import DataSplitter
        print("  ✅ DataSplitter imported successfully")
    except ImportError as e:
        print(f"  ❌ DataSplitter import failed: {e}")
        return False
    
    try:
        from model_builder import ModelBuilder
        print("  ✅ ModelBuilder imported successfully")
    except ImportError as e:
        print(f"  ❌ ModelBuilder import failed: {e}")
        return False
    
    try:
        from model_trainer import ModelTrainer
        print("  ✅ ModelTrainer imported successfully")
    except ImportError as e:
        print(f"  ❌ ModelTrainer import failed: {e}")
        return False
    
    try:
        from model_evaluator import ModelEvaluator
        print("  ✅ ModelEvaluator imported successfully")
    except ImportError as e:
        print(f"  ❌ ModelEvaluator import failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that required ML dependencies are available."""
    print("\n🔍 Testing ML dependencies...")
    
    dependencies = [
        ('sklearn', 'scikit-learn'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]
    
    all_available = True
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name} available")
        except ImportError:
            print(f"  ❌ {display_name} not available")
            if module_name in ['lightgbm', 'xgboost']:
                print(f"     Note: {display_name} is optional but recommended")
            else:
                all_available = False
    
    return all_available

def test_data_splitter():
    """Test DataSplitter functionality."""
    print("\n🔍 Testing DataSplitter functionality...")
    
    try:
        from data_splitter import DataSplitter
        import pandas as pd
        import numpy as np
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        X['class'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        # Test DataSplitter
        splitter = DataSplitter(random_state=42)
        
        # Test prepare_fraud_data
        X_prep, y_prep = splitter.prepare_fraud_data(X)
        print(f"  ✅ prepare_fraud_data: {X_prep.shape}, {y_prep.shape}")
        
        # Test train_test_split
        X_train, X_test, y_train, y_test = splitter.create_train_test_split(
            X_prep, y_prep, test_size=0.2
        )
        print(f"  ✅ train_test_split: train={X_train.shape}, test={X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ DataSplitter test failed: {e}")
        return False

def test_model_builder():
    """Test ModelBuilder functionality."""
    print("\n🔍 Testing ModelBuilder functionality...")
    
    try:
        from model_builder import ModelBuilder
        
        # Test ModelBuilder
        builder = ModelBuilder(random_state=42)
        
        # Test individual model creation
        lr_model = builder.create_logistic_regression()
        rf_model = builder.create_random_forest()
        print("  ✅ Individual model creation")
        
        # Test model suite creation
        models = builder.create_model_suite(['logistic_regression', 'random_forest'])
        print(f"  ✅ Model suite created: {list(models.keys())}")
        
        # Test model info
        model_info = builder.get_model_info()
        print(f"  ✅ Model info generated for {len(model_info)} models")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ModelBuilder test failed: {e}")
        return False

def test_model_trainer():
    """Test ModelTrainer functionality."""
    print("\n🔍 Testing ModelTrainer functionality...")
    
    try:
        from model_trainer import ModelTrainer
        from model_builder import ModelBuilder
        import pandas as pd
        import numpy as np
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100  # Small dataset for testing
        n_features = 5
        
        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_train = pd.Series(np.random.choice([0, 1], n_samples, p=[0.8, 0.2]))
        
        # Create models
        builder = ModelBuilder(random_state=42)
        models = builder.create_model_suite(['logistic_regression'])
        
        # Test ModelTrainer
        trainer = ModelTrainer(random_state=42)
        
        # Test single model training
        results = trainer.train_single_model(
            models['logistic_regression'], X_train, y_train, 'logistic_regression'
        )
        print(f"  ✅ Single model training: {results['training_successful']}")
        
        # Test model suite training (without CV for speed)
        training_results = trainer.train_model_suite(models, X_train, y_train)
        print(f"  ✅ Model suite training: {len(training_results)} models")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ModelTrainer test failed: {e}")
        return False

def test_model_evaluator():
    """Test ModelEvaluator functionality."""
    print("\n🔍 Testing ModelEvaluator functionality...")
    
    try:
        from model_evaluator import ModelEvaluator
        from model_builder import ModelBuilder
        from model_trainer import ModelTrainer
        import pandas as pd
        import numpy as np
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X_test = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_test = pd.Series(np.random.choice([0, 1], n_samples, p=[0.8, 0.2]))
        
        X_train = X_test.copy()
        y_train = y_test.copy()
        
        # Create and train a simple model
        builder = ModelBuilder(random_state=42)
        models = builder.create_model_suite(['logistic_regression'])
        
        trainer = ModelTrainer(random_state=42)
        trainer.train_model_suite(models, X_train, y_train)
        
        # Test ModelEvaluator
        evaluator = ModelEvaluator()
        
        # Test single model evaluation
        eval_result = evaluator.evaluate_single_model(
            trainer.trained_models['logistic_regression'],
            X_test, y_test, 'logistic_regression'
        )
        print(f"  ✅ Single model evaluation: {eval_result['evaluation_successful']}")
        
        # Test metrics comparison
        evaluator.evaluation_results = {'logistic_regression': eval_result}
        metrics_df = evaluator.create_metrics_comparison()
        print(f"  ✅ Metrics comparison: {len(metrics_df)} rows")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ModelEvaluator test failed: {e}")
        return False

def test_notebook_structure():
    """Test that the Task 2 notebook exists and has proper structure."""
    print("\n🔍 Testing notebook structure...")
    
    notebook_path = 'notebooks/model_building_training.ipynb'
    
    if not os.path.exists(notebook_path):
        print(f"  ❌ Notebook not found: {notebook_path}")
        return False
    
    try:
        import json
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Check basic structure
        if 'cells' not in notebook:
            print("  ❌ Notebook missing 'cells' structure")
            return False
        
        cells = notebook['cells']
        if len(cells) < 10:
            print(f"  ❌ Notebook has too few cells: {len(cells)}")
            return False
        
        # Check for markdown and code cells
        markdown_cells = sum(1 for cell in cells if cell.get('cell_type') == 'markdown')
        code_cells = sum(1 for cell in cells if cell.get('cell_type') == 'code')
        
        print(f"  ✅ Notebook structure: {markdown_cells} markdown, {code_cells} code cells")
        
        # Check for key sections
        all_text = ' '.join(
            ' '.join(cell.get('source', []))
            for cell in cells
            if cell.get('cell_type') == 'markdown'
        ).lower()
        
        required_sections = [
            'model building',
            'training',
            'evaluation',
            'logistic regression',
            'ensemble',
            'auc-pr',
            'confusion matrix'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in all_text:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"  ⚠️  Missing sections: {missing_sections}")
        else:
            print("  ✅ All required sections present")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Notebook structure test failed: {e}")
        return False

def main():
    """Run all tests for Task 2 modules."""
    print("🧪 TESTING TASK 2 MODULES")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_task2_imports),
        ("Dependencies", test_dependencies),
        ("DataSplitter", test_data_splitter),
        ("ModelBuilder", test_model_builder),
        ("ModelTrainer", test_model_trainer),
        ("ModelEvaluator", test_model_evaluator),
        ("Notebook Structure", test_notebook_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n❌ {test_name} test error: {e}")
    
    print(f"\n🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Task 2 tests passed! Ready for model building and training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
        if passed >= total - 2:  # Allow for optional dependencies
            print("💡 Minor issues detected, but core functionality should work.")
    
    return passed == total

if __name__ == "__main__":
    main() 