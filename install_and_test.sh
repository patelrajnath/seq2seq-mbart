#!/bin/bash

# Portable installation and testing script for mBART implementation
# Works without sudo privileges

set -e

echo "🚀 Setting up mBART environment for English-Romanian translation..."
echo

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create local installation directory
INSTALL_DIR="local_python_packages"
mkdir -p "$INSTALL_DIR"

# Install packages to local directory
echo "📦 Installing Python packages locally..."
python3 -m pip install --user --upgrade pip setuptools wheel

# Install required packages
echo "🔧 Installing dependencies..."
python3 -m pip install --user \
    torch \
    transformers \
    datasets \
    sentencepiece \
    sacrebleu \
    rouge-score \
    wandb \
    numpy \
    pandas \
    tqdm

echo "✅ Dependencies installed successfully!"
echo

# Test installation
echo "🧪 Testing installation..."
cat << 'EOF' > test_imports.py
import sys
print("Python version:", sys.version)

try:
    import torch
    print("✓ PyTorch version:", torch.__version__)
    print("✓ CUDA available:", torch.cuda.is_available())
    
    import transformers
    print("✓ Transformers version:", transformers.__version__)
    
    import datasets
    print("✓ Datasets version:", datasets.__version__)
    
    import sentencepiece
    print("✓ SentencePiece available")
    
    import sacrebleu
    print("✓ SacreBLEU available")
    
    import rouge_score
    print("✓ ROUGE score available")
    
    print("\n🎉 All dependencies installed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
EOF

python3 test_imports.py

echo
echo "🎯 Ready to run training!"
echo "Run the following commands:"
echo
# Create a simple test with dummy data
cat << 'EOF' > quick_test.py
from src.data import DataProcessor
from src.model import MultilingualTranslationModel
import torch

print("🔍 Testing data loading...")
processor = DataProcessor()
try:
    data = processor.load_wmt_en_ro("train")[:5]
    print(f"✓ Loaded {len(data)} sample translation pairs")
    for i, item in enumerate(data[:3]):
        print(f"  {i+1}. {item['source']} → {item['target']}")
except Exception as e:
    print(f"⚠️  Using dummy data: {e}")
    data = processor._create_dummy_data()
    for item in data:
        print(f"  {item['source']} → {item['target']}")

print("\n🔍 Testing model loading...")
model = MultilingualTranslationModel()
print("✓ Model loaded successfully")
print(f"✓ Model device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

print("\n🚀 Ready to start training!")
print("Run: python3 scripts/train.py --mode finetune --batch_size 4 --max_epochs 1")
EOF

python3 quick_test.py

echo
echo "📝 Created test files:"
echo "  - test_imports.py: Dependency verification"
echo "  - quick_test.py: Quick functionality test"
echo
echo "✨ Setup complete! You can now:"
echo "  1. Test with: python3 quick_test.py"
echo "  2. Train with: python3 scripts/train.py --mode finetune --batch_size 4 --max_epochs 1"
echo "  3. Evaluate with: python3 scripts/train.py --mode evaluate"