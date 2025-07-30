"""
Quick diagnostic script to check the current state of notebook variables.
This helps identify which cells have been executed.
"""

def check_notebook_state():
    """Check what key variables are defined in the current environment."""
    
    print("🔍 CHECKING NOTEBOOK VARIABLE STATE")
    print("=" * 50)
    
    # List of key variables that should be defined at different stages
    variables_to_check = {
        'Setup Stage': ['pd', 'np', 'DataLoader', 'ModelBuilder'],
        'Data Loading': ['data_loader', 'fraud_data', 'creditcard_data'],
        'Data Preparation': ['data_splitter', 'datasets', 'dataset_info'],
        'Model Building': ['model_builder', 'models'],
        'Model Training': ['model_trainer', 'training_results'],
        'Model Evaluation': ['model_evaluator', 'evaluation_results']
    }
    
    print("\n📋 VARIABLE STATUS CHECK:")
    print("-" * 30)
    
    all_defined = True
    
    for stage, vars_list in variables_to_check.items():
        print(f"\n{stage}:")
        stage_complete = True
        
        for var_name in vars_list:
            try:
                # Check if variable exists in global namespace
                exec(f"temp = {var_name}")
                print(f"  ✅ {var_name} - DEFINED")
            except NameError:
                print(f"  ❌ {var_name} - NOT DEFINED")
                stage_complete = False
                all_defined = False
            except Exception as e:
                print(f"  ⚠️  {var_name} - ERROR: {e}")
                stage_complete = False
        
        if stage_complete:
            print(f"  🎯 {stage} - COMPLETE")
        else:
            print(f"  🚫 {stage} - INCOMPLETE")
    
    print(f"\n{'🎉 ALL STAGES COMPLETE!' if all_defined else '⚠️  SOME STAGES MISSING'}")
    
    # Specific check for training_results
    print(f"\n🔍 SPECIFIC CHECK FOR training_results:")
    print("-" * 35)
    
    try:
        exec("temp = training_results")
        print("✅ training_results is defined")
        
        # Check its contents
        exec("keys = list(training_results.keys())")
        exec("print(f'  Keys: {keys}')")
        
        for key in ['fraud', 'creditcard']:
            try:
                exec(f"temp = training_results['{key}']")
                print(f"  ✅ training_results['{key}'] exists")
            except KeyError:
                print(f"  ❌ training_results['{key}'] missing")
    
    except NameError:
        print("❌ training_results is NOT DEFINED")
        print("\n🔧 SOLUTION: Run the Model Training cell (cell 11) first!")
        print("   This cell contains: 'training_results = {}'")
    
    print(f"\n📝 NEXT STEPS:")
    print("-" * 15)
    
    if not all_defined:
        print("1. 🔄 Run cells in sequence starting from the beginning")
        print("2. ⚡ Or use 'Run All' to execute all cells")
        print("3. 🎯 Ensure each cell completes before running the next")
    else:
        print("✅ All variables defined - notebook is ready for evaluation!")

if __name__ == "__main__":
    # This would be run in the notebook environment
    print("Copy and paste this code into a notebook cell to check variable state:")
    print("\n" + "="*50)
    print("""
# Quick diagnostic check
variables_to_check = ['pd', 'np', 'data_loader', 'fraud_data', 'datasets', 'models', 'training_results']

for var in variables_to_check:
    try:
        exec(f"temp = {var}")
        print(f"✅ {var}")
    except NameError:
        print(f"❌ {var} - NOT DEFINED")

# Check training_results specifically
try:
    print(f"training_results keys: {list(training_results.keys())}")
except NameError:
    print("❌ Need to run Model Training cell first!")
    """) 