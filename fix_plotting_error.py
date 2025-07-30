import json

def fix_plotting_error(notebook_path):
    """Fix the plotting error by removing unsupported ax and title parameters"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the cell with the problematic plotting code
    target_cell_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'plot_roc_curves(ax=' in source and 'plt.subplots' in source:
                target_cell_idx = i
                break
    
    if target_cell_idx is None:
        print("Could not find the problematic plotting cell")
        return False
    
    print(f"Found problematic cell at index {target_cell_idx}")
    
    # Replace the problematic plotting code
    fixed_source = [
        "# Plot confusion matrices for best models\n",
        "print(\"📊 PLOTTING MODEL PERFORMANCE VISUALIZATIONS\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "# Quick fix: Create model_evaluator if not defined\n",
        "if 'model_evaluator' not in globals():\n",
        "    model_evaluator = ModelEvaluator(figsize=(12, 8))\n",
        "\n",
        "# Plot confusion matrices for fraud detection\n",
        "print(\"\\n🎯 FRAUD DETECTION - Confusion Matrices\")\n",
        "model_evaluator.plot_confusion_matrices()\n",
        "\n",
        "# Plot confusion matrices for credit card  \n",
        "print(\"\\n💳 CREDIT CARD - Confusion Matrices\")\n",
        "cc_evaluator.plot_confusion_matrices()\n",
        "\n",
        "# Plot ROC and Precision-Recall curves\n",
        "print(\"\\n📈 ROC and Precision-Recall Curves\")\n",
        "\n",
        "# Fraud detection curves\n",
        "print(\"\\n🎯 Fraud Detection Dataset\")\n",
        "model_evaluator.plot_roc_curves()\n",
        "model_evaluator.plot_precision_recall_curves()\n",
        "\n",
        "# Credit card curves\n",
        "print(\"\\n💳 Credit Card Dataset\")\n",
        "cc_evaluator.plot_roc_curves()\n",
        "cc_evaluator.plot_precision_recall_curves()\n",
        "\n",
        "# Get model recommendations\n",
        "print(\"\\n🏆 MODEL RECOMMENDATIONS\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "fraud_recommendation = model_evaluator.generate_model_recommendation(primary_metric='auc_pr')\n",
        "cc_recommendation = cc_evaluator.generate_model_recommendation(primary_metric='auc_pr')\n",
        "\n",
        "print(\"\\n🎯 FRAUD DETECTION DATASET:\")\n",
        "print(f\"Best Model: {fraud_recommendation['best_model']}\")\n",
        "print(f\"Primary Metric (AUC-PR): {fraud_recommendation['best_score']:.4f}\")\n",
        "print(f\"Reasoning: {fraud_recommendation['reasoning']}\")\n",
        "\n",
        "print(\"\\n💳 CREDIT CARD DATASET:\")\n",
        "print(f\"Best Model: {cc_recommendation['best_model']}\")\n",
        "print(f\"Primary Metric (AUC-PR): {cc_recommendation['best_score']:.4f}\")\n",
        "print(f\"Reasoning: {cc_recommendation['reasoning']}\")\n",
        "\n",
        "# Store best models for SHAP analysis\n",
        "best_models = {\n",
        "    'fraud': {\n",
        "        'model': fraud_trainer.trained_models[fraud_recommendation['best_model']],\n",
        "        'model_name': fraud_recommendation['best_model'],\n",
        "        'X_train': X_train_fraud,\n",
        "        'X_test': X_test_fraud,\n",
        "        'y_test': y_test_fraud\n",
        "    },\n",
        "    'creditcard': {\n",
        "        'model': cc_trainer.trained_models[cc_recommendation['best_model']],\n",
        "        'model_name': cc_recommendation['best_model'],\n",
        "        'X_train': X_train_cc,\n",
        "        'X_test': X_test_cc,\n",
        "        'y_test': y_test_cc\n",
        "    }\n",
        "}\n",
        "\n",
        "print(\"\\n✅ Model evaluation completed! Best models identified for SHAP analysis.\")\n"
    ]
    
    # Update the cell source
    notebook['cells'][target_cell_idx]['source'] = fixed_source
    
    # Clear outputs to remove any cached errors
    notebook['cells'][target_cell_idx]['outputs'] = []
    notebook['cells'][target_cell_idx]['execution_count'] = None
    
    # Write the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Plotting error fix applied!")
    return True

if __name__ == "__main__":
    notebook_path = "notebooks/model_building_training.ipynb"
    success = fix_plotting_error(notebook_path)
    if success:
        print("🎉 Plotting error fixed! The methods now call plot_roc_curves() and plot_precision_recall_curves() without unsupported parameters.")
        print("\n📋 What was fixed:")
        print("• Removed unsupported 'ax' and 'title' parameters from plotting methods")
        print("• Simplified plotting to use individual plots instead of subplots")
        print("• Added quick fix for model_evaluator if not defined")
        print("• Methods now create their own plots individually")
    else:
        print("❌ Failed to fix the plotting error") 