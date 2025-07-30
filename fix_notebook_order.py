"""
Script to fix the notebook cell order for proper execution sequence.
The cells got mixed up when we added the SHAP analysis content.
"""

import json
import os

def fix_notebook_cell_order():
    """Fix the cell order in the notebook for proper execution."""
    
    notebook_path = 'notebooks/model_building_training.ipynb'
    
    # Read existing notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    
    # Define the correct order of cells by identifying them by content
    ordered_cells = []
    
    # 1. Task 2 header (raw)
    for cell in cells:
        if cell.get('cell_type') == 'raw' and 'Task 2: Model Building and Training' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 2. Setup and imports header (raw)
    for cell in cells:
        if cell.get('cell_type') == 'raw' and '## 1. Setup and Imports' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 3. Setup and imports code
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'sys.path.append' in str(cell.get('source', '')) and 
            'DataLoader' in str(cell.get('source', ''))):
            ordered_cells.append(cell)
            break
    
    # 4. Data loading header (raw)
    for cell in cells:
        if cell.get('cell_type') == 'raw' and '## 2. Data Loading' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 5. Data loading code
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'data_loader = DataLoader' in str(cell.get('source', ''))):
            ordered_cells.append(cell)
            break
    
    # 6. Data preparation header (raw)
    for cell in cells:
        if cell.get('cell_type') == 'raw' and '## 3. Data Preparation and Train-Test Split' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 7. Data preparation code
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'data_splitter = DataSplitter' in str(cell.get('source', ''))):
            ordered_cells.append(cell)
            break
    
    # 8. Model building header (raw)
    for cell in cells:
        if cell.get('cell_type') == 'raw' and '## 4. Model Building' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 9. Model building code
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'model_builder = ModelBuilder' in str(cell.get('source', ''))):
            ordered_cells.append(cell)
            break
    
    # 10. Model training header (raw)
    for cell in cells:
        if cell.get('cell_type') == 'raw' and '## 5. Model Training and Cross-Validation' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 11. Model training code
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'model_trainer = ModelTrainer' in str(cell.get('source', '')) and
            'training_results = {}' in str(cell.get('source', ''))):
            ordered_cells.append(cell)
            break
    
    # 12. Model evaluation header (raw)
    for cell in cells:
        if cell.get('cell_type') == 'raw' and '## 6. Model Evaluation on Test Sets' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 13. Model evaluator initialization
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'model_evaluator = ModelEvaluator' in str(cell.get('source', '')) and
            len(str(cell.get('source', '')).strip().split('\n')) <= 5):  # Simple initialization only
            ordered_cells.append(cell)
            break
    
    # 14. Model evaluation code (the one with training_results)
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'training_results[' in str(cell.get('source', '')) and
            'EVALUATING MODELS ON TEST SETS' in str(cell.get('source', ''))):
            ordered_cells.append(cell)
            break
    
    # 15. Model performance visualization header
    for cell in cells:
        if cell.get('cell_type') == 'raw' and '## 7. Model Performance Visualization' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 16. Model performance visualization code
    for cell in cells:
        if (cell.get('cell_type') == 'code' and 
            'PLOTTING MODEL PERFORMANCE VISUALIZATIONS' in str(cell.get('source', ''))):
            ordered_cells.append(cell)
            break
    
    # 17. SHAP header
    for cell in cells:
        if cell.get('cell_type') == 'raw' and 'Task 3 - Model Explainability with SHAP' in str(cell.get('source', '')):
            ordered_cells.append(cell)
            break
    
    # 18-27. All remaining SHAP cells in order
    shap_cells = []
    for cell in cells:
        if cell not in ordered_cells:
            # Check if it's a SHAP-related cell
            source_str = str(cell.get('source', ''))
            if any(keyword in source_str for keyword in ['SHAP', 'explainer', 'ModelExplainer', 'shap_values', 'Global Feature Importance', 'Local Explanations', 'Feature Dependence', 'Fraud Driver Analysis', 'Final SHAP']):
                shap_cells.append(cell)
    
    # Sort SHAP cells by their content order
    shap_order = [
        'ModelExplainer',  # SHAP setup
        'CALCULATING SHAP VALUES',  # SHAP calculation  
        'Global Feature Importance',  # Global analysis header
        'GENERATING SHAP SUMMARY PLOTS',  # Global analysis code
        'Local Explanations',  # Local analysis header
        'GENERATING LOCAL EXPLANATIONS',  # Local analysis code
        'Feature Dependence',  # Feature analysis header
        'FEATURE DEPENDENCE AND INTERACTION',  # Feature analysis code
        'Fraud Driver Analysis',  # Fraud analysis header
        'COMPREHENSIVE FRAUD DRIVER ANALYSIS',  # Fraud analysis code
        'Final SHAP',  # Final reports header
        'GENERATING COMPREHENSIVE SHAP REPORTS'  # Final reports code
    ]
    
    for keyword in shap_order:
        for cell in shap_cells:
            if keyword in str(cell.get('source', '')) and cell not in ordered_cells:
                ordered_cells.append(cell)
                break
    
    # Add any remaining cells
    for cell in cells:
        if cell not in ordered_cells:
            ordered_cells.append(cell)
    
    # Update notebook with reordered cells
    notebook['cells'] = ordered_cells
    
    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Fixed notebook cell order in {notebook_path}")
    print(f"   Total cells: {len(ordered_cells)}")
    print(f"   Execution order now correct!")
    
    # Print the new order for verification
    print("\n📋 NEW CELL ORDER:")
    for i, cell in enumerate(ordered_cells):
        cell_type = cell.get('cell_type', 'unknown')
        source = str(cell.get('source', ''))[:60].replace('\n', ' ')
        print(f"  {i+1:2d}. [{cell_type:4s}] {source}...")

if __name__ == "__main__":
    fix_notebook_cell_order() 