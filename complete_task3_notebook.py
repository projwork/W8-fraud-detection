"""
Script to complete the Task 2 notebook with Task 3 SHAP analysis cells.
This adds the remaining cells for model explainability using SHAP.
"""

import json
import os

def add_shap_cells_to_notebook():
    """Add SHAP analysis cells to the existing notebook."""
    
    notebook_path = 'notebooks/model_building_training.ipynb'
    
    # Read existing notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Additional cells for SHAP analysis
    additional_cells = [
        # Cell for performance visualization
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot confusion matrices for best models\n",
                "print(\"📊 PLOTTING MODEL PERFORMANCE VISUALIZATIONS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# Plot confusion matrices for fraud detection\n",
                "print(\"\\n🎯 FRAUD DETECTION - Confusion Matrices\")\n",
                "model_evaluator.plot_confusion_matrices()\n",
                "\n",
                "# Plot confusion matrices for credit card  \n",
                "print(\"\\n💳 CREDIT CARD - Confusion Matrices\")\n",
                "cc_evaluator.plot_confusion_matrices()\n",
                "\n",
                "# Plot ROC curves\n",
                "print(\"\\n📈 ROC and Precision-Recall Curves\")\n",
                "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))\n",
                "\n",
                "# Fraud detection curves\n",
                "model_evaluator.plot_roc_curves(ax=ax1, title=\"Fraud Detection - ROC Curves\")\n",
                "model_evaluator.plot_precision_recall_curves(ax=ax2, title=\"Fraud Detection - PR Curves\")\n",
                "\n",
                "# Credit card curves\n",
                "cc_evaluator.plot_roc_curves(ax=ax3, title=\"Credit Card - ROC Curves\")\n",
                "cc_evaluator.plot_precision_recall_curves(ax=ax4, title=\"Credit Card - PR Curves\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
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
                "print(\"\\n✅ Model evaluation completed! Best models identified for SHAP analysis.\")"
            ]
        },
        # Task 3 header
        {
            "cell_type": "raw",
            "metadata": {
                "vscode": {
                    "languageId": "raw"
                }
            },
            "source": [
                "## 8. Task 3 - Model Explainability with SHAP\n",
                "\n",
                "### SHAP (Shapley Additive exPlanations) Analysis\n",
                "\n",
                "SHAP provides unified measure of feature importance and allows us to understand:\n",
                "- **Global Interpretability**: Which features are most important overall\n",
                "- **Local Interpretability**: How features contribute to individual predictions\n",
                "- **Feature Interactions**: How features work together\n",
                "\n",
                "We'll analyze both the best-performing models to understand the key drivers of fraud detection."
            ]
        },
        # SHAP setup
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import SHAP explainer module\n",
                "from model_explainer import ModelExplainer\n",
                "\n",
                "# Install SHAP if not available\n",
                "try:\n",
                "    import shap\n",
                "    print(\"✅ SHAP is available\")\n",
                "except ImportError:\n",
                "    print(\"Installing SHAP...\")\n",
                "    !pip install shap\n",
                "    import shap\n",
                "    print(\"✅ SHAP installed and imported\")\n",
                "\n",
                "print(\"\\n🔍 INITIALIZING SHAP EXPLAINERS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# Initialize SHAP explainers for best models\n",
                "explainers = {}\n",
                "\n",
                "for dataset_name, model_info in best_models.items():\n",
                "    print(f\"\\n📊 Setting up SHAP explainer for {dataset_name.upper()} dataset...\")\n",
                "    print(f\"   Model: {model_info['model_name']}\")\n",
                "    print(f\"   Features: {model_info['X_train'].shape[1]}\")\n",
                "    \n",
                "    explainer = ModelExplainer(\n",
                "        model=model_info['model'],\n",
                "        X_train=model_info['X_train'],\n",
                "        X_test=model_info['X_test'],\n",
                "        feature_names=list(model_info['X_train'].columns) if hasattr(model_info['X_train'], 'columns') else None\n",
                "    )\n",
                "    \n",
                "    explainers[dataset_name] = {\n",
                "        'explainer': explainer,\n",
                "        'model_name': model_info['model_name'],\n",
                "        'y_test': model_info['y_test']\n",
                "    }\n",
                "    \n",
                "    print(f\"   ✅ SHAP explainer initialized\")\n",
                "\n",
                "print(\"\\n🎯 SHAP explainers ready for both datasets!\")"
            ]
        },
        # SHAP calculation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate SHAP values\n",
                "print(\"⚙️ CALCULATING SHAP VALUES\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "shap_results = {}\n",
                "\n",
                "for dataset_name, explainer_info in explainers.items():\n",
                "    print(f\"\\n📊 Calculating SHAP values for {dataset_name.upper()} dataset...\")\n",
                "    \n",
                "    explainer = explainer_info['explainer']\n",
                "    \n",
                "    # Calculate SHAP values for a sample of test data (for performance)\n",
                "    sample_size = min(1000, len(explainer.X_test))\n",
                "    print(f\"   Using sample size: {sample_size}\")\n",
                "    \n",
                "    start_time = time.time()\n",
                "    shap_values = explainer.calculate_shap_values(sample_size=sample_size, on_test=True)\n",
                "    calculation_time = time.time() - start_time\n",
                "    \n",
                "    print(f\"   ✅ SHAP values calculated in {calculation_time:.2f} seconds\")\n",
                "    print(f\"   Shape: {shap_values.shape}\")\n",
                "    \n",
                "    shap_results[dataset_name] = {\n",
                "        'explainer': explainer,\n",
                "        'shap_values': shap_values,\n",
                "        'model_name': explainer_info['model_name'],\n",
                "        'y_test': explainer_info['y_test']\n",
                "    }\n",
                "\n",
                "print(\"\\n🎯 SHAP value calculation completed for all models!\")"
            ]
        },
        # Global interpretability
        {
            "cell_type": "raw",
            "metadata": {
                "vscode": {
                    "languageId": "raw"
                }
            },
            "source": [
                "### 8.1 Global Feature Importance - SHAP Summary Plots"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate SHAP Summary Plots for Global Interpretability\n",
                "print(\"📊 GENERATING SHAP SUMMARY PLOTS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "for dataset_name, shap_info in shap_results.items():\n",
                "    explainer = shap_info['explainer']\n",
                "    model_name = shap_info['model_name']\n",
                "    \n",
                "    print(f\"\\n🎯 {dataset_name.upper()} Dataset - {model_name.upper()} Model\")\n",
                "    print(\"-\" * 50)\n",
                "    \n",
                "    # Summary plot (dot plot) - shows feature importance and feature effects\n",
                "    print(\"📈 SHAP Summary Plot (Feature Importance & Effects)\")\n",
                "    explainer.plot_summary(\n",
                "        plot_type='dot',\n",
                "        max_display=15,\n",
                "        title=f'{dataset_name.title()} - {model_name.title()} SHAP Summary'\n",
                "    )\n",
                "    \n",
                "    # Bar plot - shows feature importance only\n",
                "    print(\"📊 SHAP Summary Plot (Feature Importance Only)\")\n",
                "    explainer.plot_summary(\n",
                "        plot_type='bar',\n",
                "        max_display=15,\n",
                "        title=f'{dataset_name.title()} - {model_name.title()} Feature Importance'\n",
                "    )\n",
                "\n",
                "print(\"\\n✅ Global SHAP analysis completed!\")"
            ]
        },
        # Local explanations
        {
            "cell_type": "raw",
            "metadata": {
                "vscode": {
                    "languageId": "raw"
                }
            },
            "source": [
                "### 8.2 Local Explanations - SHAP Force Plots and Waterfall Plots"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate local explanations for individual predictions\n",
                "print(\"🔍 GENERATING LOCAL EXPLANATIONS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# For each dataset, show explanations for fraud and non-fraud cases\n",
                "for dataset_name, shap_info in shap_results.items():\n",
                "    explainer = shap_info['explainer']\n",
                "    model_name = shap_info['model_name']\n",
                "    y_test_sample = shap_info['y_test']\n",
                "    \n",
                "    print(f\"\\n🎯 {dataset_name.upper()} Dataset - Local Explanations\")\n",
                "    print(\"-\" * 50)\n",
                "    \n",
                "    # Find fraud and non-fraud cases\n",
                "    if hasattr(y_test_sample, 'iloc'):\n",
                "        fraud_indices = np.where(y_test_sample.iloc[:len(shap_info['shap_values'])] == 1)[0]\n",
                "        non_fraud_indices = np.where(y_test_sample.iloc[:len(shap_info['shap_values'])] == 0)[0]\n",
                "    else:\n",
                "        fraud_indices = np.where(y_test_sample[:len(shap_info['shap_values'])] == 1)[0]\n",
                "        non_fraud_indices = np.where(y_test_sample[:len(shap_info['shap_values'])] == 0)[0]\n",
                "    \n",
                "    # Show examples if available\n",
                "    if len(fraud_indices) > 0:\n",
                "        fraud_idx = fraud_indices[0]\n",
                "        print(f\"🚨 FRAUD CASE EXPLANATION (Index: {fraud_idx})\")\n",
                "        \n",
                "        # Force plot\n",
                "        explainer.plot_force_plot(fraud_idx, matplotlib=True)\n",
                "        \n",
                "        # Waterfall plot\n",
                "        try:\n",
                "            explainer.plot_waterfall(fraud_idx)\n",
                "        except:\n",
                "            print(\"   Waterfall plot not available for this model type\")\n",
                "    \n",
                "    if len(non_fraud_indices) > 0:\n",
                "        non_fraud_idx = non_fraud_indices[0]\n",
                "        print(f\"✅ NON-FRAUD CASE EXPLANATION (Index: {non_fraud_idx})\")\n",
                "        \n",
                "        # Force plot\n",
                "        explainer.plot_force_plot(non_fraud_idx, matplotlib=True)\n",
                "        \n",
                "        # Waterfall plot  \n",
                "        try:\n",
                "            explainer.plot_waterfall(non_fraud_idx)\n",
                "        except:\n",
                "            print(\"   Waterfall plot not available for this model type\")\n",
                "\n",
                "print(\"\\n✅ Local explanations completed!\")"
            ]
        },
        # Feature dependence
        {
            "cell_type": "raw",
            "metadata": {
                "vscode": {
                    "languageId": "raw"
                }
            },
            "source": [
                "### 8.3 Feature Dependence and Interaction Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze feature dependencies and interactions\n",
                "print(\"🔗 FEATURE DEPENDENCE AND INTERACTION ANALYSIS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "for dataset_name, shap_info in shap_results.items():\n",
                "    explainer = shap_info['explainer']\n",
                "    model_name = shap_info['model_name']\n",
                "    \n",
                "    print(f\"\\n🎯 {dataset_name.upper()} Dataset - Feature Dependencies\")\n",
                "    print(\"-\" * 50)\n",
                "    \n",
                "    # Get top features for dependence analysis\n",
                "    feature_importance = explainer.get_feature_importance()\n",
                "    top_features = feature_importance.head(5)['feature'].tolist()\n",
                "    \n",
                "    print(f\"📊 Top 5 features for dependence analysis: {top_features}\")\n",
                "    \n",
                "    # Create dependence plots for top features\n",
                "    for i, feature in enumerate(top_features[:3]):  # Show top 3 to avoid too many plots\n",
                "        print(f\"\\n📈 Dependence plot for: {feature}\")\n",
                "        try:\n",
                "            explainer.plot_dependence(feature)\n",
                "        except Exception as e:\n",
                "            print(f\"   Could not create dependence plot for {feature}: {e}\")\n",
                "    \n",
                "    # Show feature importance table\n",
                "    print(f\"\\n📋 Feature Importance Rankings:\")\n",
                "    print(feature_importance.head(10).to_string(index=False))\n",
                "\n",
                "print(\"\\n✅ Feature dependence analysis completed!\")"
            ]
        },
        # Fraud analysis
        {
            "cell_type": "raw",
            "metadata": {
                "vscode": {
                    "languageId": "raw"
                }
            },
            "source": [
                "### 8.4 Fraud Driver Analysis and Insights"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Comprehensive fraud driver analysis\n",
                "print(\"🕵️ COMPREHENSIVE FRAUD DRIVER ANALYSIS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "fraud_insights = {}\n",
                "\n",
                "for dataset_name, shap_info in shap_results.items():\n",
                "    explainer = shap_info['explainer']\n",
                "    model_name = shap_info['model_name']\n",
                "    y_test_sample = shap_info['y_test']\n",
                "    \n",
                "    print(f\"\\n🎯 {dataset_name.upper()} Dataset Analysis\")\n",
                "    print(\"-\" * 50)\n",
                "    \n",
                "    # Analyze fraud drivers\n",
                "    sample_size = len(shap_info['shap_values'])\n",
                "    if hasattr(y_test_sample, 'iloc'):\n",
                "        y_sample = y_test_sample.iloc[:sample_size]\n",
                "    else:\n",
                "        y_sample = y_test_sample[:sample_size]\n",
                "    \n",
                "    analysis = explainer.analyze_fraud_drivers(y_test=y_sample, top_features=10)\n",
                "    fraud_insights[dataset_name] = analysis\n",
                "    \n",
                "    # Display key findings\n",
                "    print(\"🔍 KEY FINDINGS:\")\n",
                "    print(\"─\" * 40)\n",
                "    \n",
                "    print(\"\\n📊 TOP FEATURE IMPORTANCE:\")\n",
                "    for idx, row in analysis['top_features'].head(5).iterrows():\n",
                "        print(f\"  {idx+1}. {row['feature']}: {row['importance']:.4f}\")\n",
                "    \n",
                "    if 'fraud_drivers' in analysis:\n",
                "        print(\"\\n🚨 TOP FRAUD DRIVERS (vs Non-Fraud):\")\n",
                "        for idx, row in analysis['fraud_drivers'].head(5).iterrows():\n",
                "            direction = \"↑\" if row['fraud_contribution_diff'] > 0 else \"↓\"\n",
                "            print(f\"  {idx+1}. {row['feature']}: {direction} {abs(row['fraud_contribution_diff']):.4f}\")\n",
                "    \n",
                "    print(\"\\n💡 INSIGHTS:\")\n",
                "    for insight in analysis['overall_insights']:\n",
                "        print(f\"  • {insight}\")\n",
                "    \n",
                "    if 'fraud_specific_insights' in analysis:\n",
                "        for insight in analysis['fraud_specific_insights']:\n",
                "            print(f\"  • {insight}\")\n",
                "\n",
                "print(\"\\n✅ Fraud driver analysis completed!\")"
            ]
        },
        # Final reports
        {
            "cell_type": "raw",
            "metadata": {
                "vscode": {
                    "languageId": "raw"
                }
            },
            "source": [
                "### 8.5 Final SHAP Explanation Reports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate comprehensive explanation reports\n",
                "print(\"📝 GENERATING COMPREHENSIVE SHAP REPORTS\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "reports = {}\n",
                "\n",
                "for dataset_name, shap_info in shap_results.items():\n",
                "    explainer = shap_info['explainer']\n",
                "    model_name = shap_info['model_name']\n",
                "    y_test_sample = shap_info['y_test']\n",
                "    \n",
                "    # Generate report\n",
                "    sample_size = len(shap_info['shap_values'])\n",
                "    if hasattr(y_test_sample, 'iloc'):\n",
                "        y_sample = y_test_sample.iloc[:sample_size]\n",
                "    else:\n",
                "        y_sample = y_test_sample[:sample_size]\n",
                "    \n",
                "    report = explainer.generate_explanation_report(\n",
                "        y_test=y_sample,\n",
                "        dataset_name=f\"{dataset_name.title()} ({model_name.title()})\"\n",
                "    )\n",
                "    \n",
                "    reports[dataset_name] = report\n",
                "    \n",
                "    print(f\"\\n{report}\")\n",
                "    print(\"\\n\" + \"=\"*80)\n",
                "\n",
                "# Summary of key fraud drivers across datasets\n",
                "print(\"\\n🎯 CROSS-DATASET FRAUD DRIVER SUMMARY\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "print(\"\\n📊 Key Insights from SHAP Analysis:\")\n",
                "print(\"-\" * 40)\n",
                "\n",
                "if 'fraud' in fraud_insights and 'creditcard' in fraud_insights:\n",
                "    fraud_analysis = fraud_insights['fraud']\n",
                "    cc_analysis = fraud_insights['creditcard']\n",
                "    \n",
                "    print(\"\\n🔍 FRAUD DETECTION DATASET (E-commerce):\")\n",
                "    if 'fraud_drivers' in fraud_analysis:\n",
                "        top_fraud_driver = fraud_analysis['fraud_drivers'].iloc[0]\n",
                "        print(f\"  • Primary fraud driver: {top_fraud_driver['feature']}\")\n",
                "        print(f\"  • Impact: {top_fraud_driver['fraud_contribution_diff']:.4f}\")\n",
                "    \n",
                "    print(\"\\n🔍 CREDIT CARD DATASET (Bank transactions):\")\n",
                "    if 'fraud_drivers' in cc_analysis:\n",
                "        top_cc_driver = cc_analysis['fraud_drivers'].iloc[0]\n",
                "        print(f\"  • Primary fraud driver: {top_cc_driver['feature']}\")\n",
                "        print(f\"  • Impact: {top_cc_driver['fraud_contribution_diff']:.4f}\")\n",
                "\n",
                "print(\"\\n🎉 TASK 3 - MODEL EXPLAINABILITY COMPLETED!\")\n",
                "print(\"=\" * 60)\n",
                "print(\"✅ SHAP analysis provides comprehensive model interpretability\")\n",
                "print(\"✅ Global and local explanations generated\")\n",
                "print(\"✅ Key fraud drivers identified\")\n",
                "print(\"✅ Feature interactions analyzed\")\n",
                "print(\"✅ Actionable insights for fraud prevention strategies\")"
            ]
        }
    ]
    
    # Add new cells to the notebook
    notebook['cells'].extend(additional_cells)
    
    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Task 3 SHAP analysis added to {notebook_path}")
    print(f"   Added {len(additional_cells)} new cells")
    print(f"   Total cells: {len(notebook['cells'])}")

if __name__ == "__main__":
    add_shap_cells_to_notebook() 