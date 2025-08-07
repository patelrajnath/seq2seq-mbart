# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

### Training
```bash
# Base model training (lightweight, default)
python scripts/train.py --mode pretrain --config configs/pretrain_base_config.json
python scripts/train.py --mode finetune --config configs/finetune_base_config.json

# Large model training (production)
python scripts/train.py --mode pretrain --config configs/pretrain_config.json
python scripts/train.py --mode finetune --config configs/translation_config.json

# Quick evaluation
python scripts/train.py --mode evaluate --model_path checkpoints/finetune_base/best_translation_model.pt
```

### Testing
```bash
# Run all tests
python test_implementation.py
python test_end_to_end.py
python test_lightweight.py

# Quick verification
python verify_setup.py
```

### Utilities
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
bash setup.sh
bash install_and_test.sh

# Quick demo
python demo_en_ro.py
python demo_working.py
```

## Architecture Overview

### Core Components
- **src/model.py**: Main model implementations (MultilingualDenoisingPretraining, MultilingualTranslationModel)
- **src/trainer.py**: Training loops for pretraining and fine-tuning with config-based parameterization
- **src/data.py**: Data processing pipeline for multilingual text (WMT16, CC100)
- **src/evaluation.py**: BLEU, ROUGE, chrF++ metrics for translation evaluation

### Configuration System
- **configs/**: JSON-based configuration for model architecture and training parameters
  - `pretrain_base_config.json`: Lightweight 512d model (6 layers)
  - `finetune_base_config.json`: mbart-base-50 fine-tuning
  - `pretrain_config.json`: Large 1024d model (12 layers) 
  - `translation_config.json`: mbart-large-50 fine-tuning

### Training Pipeline
1. **Pretraining**: Denoising objective on monolingual corpora
2. **Fine-tuning**: English-Romanian translation task
3. **Evaluation**: Comprehensive metrics on test sets

## Key File Patterns

### Model Usage
```python
# Load model from config
from src.trainer import run_pretraining, run_finetuning
run_pretraining(config_path="configs/pretrain_base_config.json")

# Custom training
from src.model import MultilingualTranslationModel
model = MultilingualTranslationModel("facebook/mbart-base-50")
```

### Data Processing
```python
from src.data import DataProcessor
processor = DataProcessor()
train_loader = processor.create_pretrain_dataloaders(batch_size=16, max_length=256)
```

### Configuration Override
```python
# Override config values via CLI
python scripts/train.py --mode pretrain --config configs/pretrain_base_config.json --batch_size 8 --max_steps 1000
```

## Environment Setup
- **Dependencies**: See requirements.txt for exact versions
- **CUDA**: Automatic device detection (cuda/cpu)
- **Memory**: Base configs require ~2GB GPU, large configs ~8GB
- **Datasets**: Auto-download WMT16 and CC100 if not cached

## Common Workflows

### Development Cycle
1. Test with base config: `python scripts/train.py --mode finetune --data_size 1000`
2. Validate with lightweight tests: `python test_lightweight.py`
3. Scale to full dataset with large config

### Debugging
- Use `--data_size` parameter for quick testing
- Check `checkpoints/` for model checkpoints
- Monitor `logs/` for training progress
- Use `verify_setup.py` for environment validation