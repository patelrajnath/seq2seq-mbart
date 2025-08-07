# Configuration Files

This directory contains configuration files for different model sizes and training scenarios.

## Base Models (Lightweight)

### Pretraining
- **File**: `pretrain_base_config.json`
- **Model**: 6 encoder/decoder layers, 512 d_model, 8 attention heads
- **Use case**: Fast experimentation, resource-constrained environments
- **Memory**: ~2GB GPU memory for batch_size=16

### Finetuning
- **File**: `finetune_base_config.json`
- **Model**: Uses mbart-base-50 (smaller than mbart-large-50)
- **Use case**: Quick fine-tuning, development/testing
- **Memory**: ~1.5GB GPU memory for batch_size=16

## Large Models (Production)

### Pretraining
- **File**: `pretrain_config.json`
- **Model**: 12 encoder/decoder layers, 1024 d_model, 16 attention heads
- **Use case**: Production pretraining, maximum performance
- **Memory**: ~8GB GPU memory for batch_size=8

### Finetuning
- **File**: `translation_config.json`
- **Model**: Uses mbart-large-50 (original large model)
- **Use case**: Production fine-tuning, best performance
- **Memory**: ~6GB GPU memory for batch_size=8

## Usage Examples

### Base Model Pretraining
```bash
python scripts/train.py --mode pretrain --config configs/pretrain_base_config.json
```

### Base Model Finetuning
```bash
python scripts/train.py --mode finetune --config configs/finetune_base_config.json
```

### Large Model Training
```bash
# Pretraining
python scripts/train.py --mode pretrain --config configs/pretrain_config.json

# Finetuning
python scripts/train.py --mode finetune --config configs/translation_config.json
```

## Custom Configuration

You can override any config values via command line:
```bash
python scripts/train.py --mode pretrain --config configs/pretrain_base_config.json --batch_size 8 --learning_rate 1e-4
```

## Memory Requirements

| Model Type | Batch Size | GPU Memory | Training Time (Est.) |
|------------|------------|------------|---------------------|
| Base       | 16         | ~2GB       | 2-4 hours           |
| Large      | 8          | ~8GB       | 8-12 hours          |
| Large      | 16         | ~16GB      | 4-6 hours           |