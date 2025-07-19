"""
Setup script for fraud detection project environment.
This script activates the virtual environment and installs required packages.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def setup_environment():
    """Set up the fraud detection project environment."""
    print("🚀 Setting up Fraud Detection Project Environment")
    print("=" * 60)
    
    # Check if we're on Windows
    is_windows = os.name == 'nt'
    
    if is_windows:
        # Windows activation command
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
    else:
        # Unix/Mac activation command
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found at .venv")
        print("Please create a virtual environment first:")
        print("  python -m venv .venv")
        return False
    
    print(f"✅ Virtual environment found at {venv_path}")
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        if is_windows:
            install_cmd = f"{pip_cmd} install -r requirements.txt"
        else:
            install_cmd = f"{activate_cmd} && pip install -r requirements.txt"
        
        success = run_command(install_cmd, "Installing required packages")
        if not success:
            return False
    else:
        print("⚠️ requirements.txt not found, skipping package installation")
    
    # Create necessary directories
    directories = ['data/processed', 'notebooks', 'results', 'models']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\n✅ Environment setup complete!")
    print("\n📋 Next steps:")
    if is_windows:
        print("1. Activate the environment: .venv\\Scripts\\activate")
    else:
        print("1. Activate the environment: source .venv/bin/activate")
    print("2. Start Jupyter: jupyter notebook")
    print("3. Open notebooks/fraud_detection_analysis.ipynb")
    print("\n🎯 Happy fraud detection analysis!")
    
    return True

if __name__ == "__main__":
    setup_environment() 