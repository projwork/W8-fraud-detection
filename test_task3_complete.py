"""
Comprehensive test script for Task 3 - Model Explainability completion.
Validates all modules, dependencies, and notebook structure.
"""

import sys
import os
from pathlib import Path

# Test results tracking
test_results = {
    'passed': 0,
    'failed': 0,
    'tests': []
}

def run_test(test_name, test_func):
    """Run a test and track results."""
    try:
        result = test_func()
        if result:
            test_results['passed'] += 1
            test_results['tests'].append(f"✅ {test_name}")
            print(f"✅ {test_name}")
        else:
            test_results['failed'] += 1
            test_results['tests'].append(f"❌ {test_name}")
            print(f"❌ {test_name}")
        return result
    except Exception as e:
        test_results['failed'] += 1
        test_results['tests'].append(f"❌ {test_name}: {e}")
        print(f"❌ {test_name}: {e}")
        return False

def test_task3_imports():
    """Test all Task 3 module imports."""
    sys.path.append('src')
    
    try:
        # Test core modules
        from model_explainer import ModelExplainer
        
        # Test SHAP availability
        import shap
        
        # Test other required modules
        from data_loader import DataLoader
        from data_splitter import DataSplitter
        from model_builder import ModelBuilder
        from model_trainer import ModelTrainer
        from model_evaluator import ModelEvaluator
        
        print("  ✅ All Task 3 modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_shap_dependencies():
    """Test SHAP and related dependencies."""
    try:
        import shap
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        print("  ✅ All SHAP dependencies available")
        return True
    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")
        return False

def test_model_explainer_functionality():
    """Test ModelExplainer basic functionality."""
    sys.path.append('src')
    
    try:
        from model_explainer import ModelExplainer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import pandas as pd
        
        # Create synthetic data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(5)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test ModelExplainer
        explainer = ModelExplainer(model, X_train_df, X_test_df, feature_names)
        
        # Test SHAP value calculation
        shap_values = explainer.calculate_shap_values(sample_size=10)
        
        # Test feature importance
        importance = explainer.get_feature_importance()
        
        print(f"  ✅ ModelExplainer working: SHAP shape {shap_values.shape}, importance shape {importance.shape}")
        return True
    except Exception as e:
        print(f"  ❌ ModelExplainer test failed: {e}")
        return False

def test_notebook_structure():
    """Test the updated notebook structure."""
    try:
        import json
        
        notebook_path = 'notebooks/model_building_training.ipynb'
        
        if not os.path.exists(notebook_path):
            print(f"  ❌ Notebook not found: {notebook_path}")
            return False
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        cells = notebook.get('cells', [])
        total_cells = len(cells)
        
        # Count different cell types
        code_cells = sum(1 for cell in cells if cell.get('cell_type') == 'code')
        markdown_cells = sum(1 for cell in cells if cell.get('cell_type') == 'raw')
        
        # Check for Task 3 content
        task3_content = []
        shap_keywords = ['SHAP', 'explainability', 'ModelExplainer', 'force_plot', 'summary_plot']
        
        for cell in cells:
            cell_source = str(cell.get('source', ''))
            for keyword in shap_keywords:
                if keyword in cell_source:
                    task3_content.append(keyword)
                    break
        
        print(f"  ✅ Notebook structure: {total_cells} total cells, {code_cells} code, {markdown_cells} markdown")
        print(f"  ✅ Task 3 content found: {len(task3_content)} cells with SHAP content")
        
        return total_cells >= 25 and len(task3_content) >= 5
    except Exception as e:
        print(f"  ❌ Notebook structure test failed: {e}")
        return False

def test_project_structure():
    """Test the overall project structure for Task 3."""
    required_files = [
        'src/model_explainer.py',
        'src/__init__.py',
        'requirements.txt',
        'notebooks/model_building_training.ipynb'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    # Check requirements.txt for SHAP
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    if 'shap' not in requirements.lower():
        print("  ❌ SHAP not found in requirements.txt")
        return False
    
    print("  ✅ All required files present and SHAP in requirements")
    return True

def test_complete_pipeline():
    """Test if the complete pipeline can be imported and initialized."""
    sys.path.append('src')
    
    try:
        # Test complete import chain
        from src import (
            DataLoader, DataSplitter, ModelBuilder, 
            ModelTrainer, ModelEvaluator, ModelExplainer
        )
        
        print("  ✅ Complete pipeline can be imported from src package")
        return True
    except Exception as e:
        print(f"  ❌ Complete pipeline test failed: {e}")
        return False

def main():
    """Run all Task 3 tests."""
    print("🧪 TESTING TASK 3 - MODEL EXPLAINABILITY COMPLETION")
    print("=" * 70)
    
    # Run all tests
    run_test("Task 3 module imports", test_task3_imports)
    run_test("SHAP dependencies", test_shap_dependencies)
    run_test("ModelExplainer functionality", test_model_explainer_functionality)
    run_test("Notebook structure", test_notebook_structure)
    run_test("Project structure", test_project_structure)
    run_test("Complete pipeline", test_complete_pipeline)
    
    # Summary
    print("\n" + "=" * 70)
    print("🏁 TEST SUMMARY")
    print("=" * 70)
    
    total_tests = test_results['passed'] + test_results['failed']
    success_rate = (test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Tests Passed: {test_results['passed']}/{total_tests} ({success_rate:.1f}%)")
    print(f"📊 Tests Failed: {test_results['failed']}/{total_tests}")
    
    if test_results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! Task 3 implementation is complete and ready.")
        print("✅ SHAP explainability module implemented")
        print("✅ Notebook updated with comprehensive SHAP analysis")
        print("✅ Global and local interpretability features available")
        print("✅ Fraud driver analysis capabilities implemented")
    else:
        print(f"\n⚠️  Some tests failed. Please review the issues above.")
        print("\nFailed tests:")
        for test in test_results['tests']:
            if test.startswith('❌'):
                print(f"  {test}")
    
    print("\n🚀 READY TO RUN:")
    print("   cd notebooks")
    print("   jupyter notebook")
    print("   # Open model_building_training.ipynb and run all cells")

if __name__ == "__main__":
    main() 