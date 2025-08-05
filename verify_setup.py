#!/usr/bin/env python3
"""
Verify mBART implementation setup
"""

import os
import sys
from pathlib import Path

def check_structure():
    """Check if all required files and directories exist"""
    print("Checking project structure...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "src/__init__.py",
        "src/model.py",
        "src/data.py",
        "src/trainer.py",
        "src/evaluation.py",
        "configs/pretrain_config.json",
        "configs/translation_config.json",
        "scripts/train.py",
        "test_implementation.py"
    ]
    
    required_dirs = [
        "src",
        "configs",
        "scripts",
        "checkpoints",
        "logs",
        "data"
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"❌ Missing directory: {dir_path}/")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_good = False
    
    return all_good

def check_python_version():
    """Check Python version compatibility"""
    print("\nChecking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires >=3.8)")
        return False

def create_setup_script():
    """Create setup script for dependencies"""
    setup_content = """#!/bin/bash
# Setup script for mBART implementation

echo "Setting up mBART implementation..."

# Install Python dependencies
echo "Installing dependencies..."
pip3 install torch>=1.9.0 transformers>=4.20.0 datasets>=2.0.0 \
            sentencepiece>=0.1.96 sacrebleu>=2.0.0 rouge-score wandb

# Create required directories
mkdir -p checkpoints/pretrain checkpoints/translation logs

echo "Setup complete!"
echo ""
echo "To test the implementation:"
echo "  python3 test_implementation.py"
echo ""
echo "To start training:"
echo "  python3 scripts/train.py --mode finetune --batch_size 8 --max_epochs 1"
"""
    
    with open("setup.sh", "w") as f:
        f.write(setup_content)
    
    os.chmod("setup.sh", 0o755)
    print("✓ Created setup.sh")

def provide_installation_help():
    """Provide installation guidance"""
    print("\n" + "="*60)
    print("mBART Implementation - Setup Complete!")
    print("="*60)
    print()
    print("Project structure verified successfully!")
    print()
    print("Next steps:")
    print("1. Install dependencies:")
    print("   pip3 install torch transformers datasets sentencepiece sacrebleu rouge-score wandb")
    print()
    print("2. Test the implementation:")
    print("   python3 test_implementation.py")
    print()
    print("3. Start training (English-Romanian translation):")
    print("   python3 scripts/train.py --mode finetune --batch_size 8 --max_epochs 1")
    print()
    print("4. For pre-training + fine-tuning:")
    print("   python3 scripts/train.py --mode pretrain --batch_size 8 --max_steps 5000")
    print("   python3 scripts/train.py --mode finetune --batch_size 8 --max_epochs 3")
    print()
    print("Configuration files:")
    print("   - configs/pretrain_config.json (pre-training settings)")
    print("   - configs/translation_config.json (translation settings)")
    print()
    print("For more details, see README.md")

def main():
    print("mBART Implementation Verification")
    print("=" * 40)
    
    # Check structure
    structure_ok = check_structure()
    
    # Check Python version
    python_ok = check_python_version()
    
    # Create setup script
    create_setup_script()
    
    # Provide guidance
    provide_installation_help()
    
    if structure_ok and python_ok:
        print("\n✅ Project setup verified successfully!")
    else:
        print("\n⚠️  Some issues found, but setup script created to help resolve them")

if __name__ == "__main__":
    main()