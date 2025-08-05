#!/bin/bash
# Setup script for mBART implementation

echo "Setting up mBART implementation..."

# Install Python dependencies
echo "Installing dependencies..."
pip3 install torch>=1.9.0 transformers>=4.20.0 datasets>=2.0.0             sentencepiece>=0.1.96 sacrebleu>=2.0.0 rouge-score wandb

# Create required directories
mkdir -p checkpoints/pretrain checkpoints/translation logs

echo "Setup complete!"
echo ""
echo "To test the implementation:"
echo "  python3 test_implementation.py"
echo ""
echo "To start training:"
echo "  python3 scripts/train.py --mode finetune --batch_size 8 --max_epochs 1"
