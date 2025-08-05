# Multilingual Denoising Pre-training for Neural Machine Translation

This repository implements the mBART model as described in the paper "Multilingual Denoising Pre-training for Neural Machine Translation" by Liu et al. (2020), specifically focusing on English-Romanian translation.

## Overview

The implementation includes:
- **Multilingual Denoising Pre-training**: Large-scale pre-training on monolingual corpora
- **Neural Machine Translation**: Fine-tuning for English-Romanian translation
- **Comprehensive Evaluation**: BLEU, ROUGE, chrF++, and exact match metrics

## Architecture Details

Based on the original mBART paper:
- **Model Size**: 680M parameters
- **Layers**: 12 encoder layers, 12 decoder layers
- **Hidden Size**: 1024 dimensions
- **Attention Heads**: 16 heads per layer
- **Vocabulary**: 250K subword tokens using SentencePiece
- **Languages**: English (en_XX) and Romanian (ro_RO)

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install with conda
conda create -n mbart python=3.8
conda activate mbart
pip install torch transformers datasets sentencepiece sacrebleu rouge-score wandb
```

### 2. Data Preparation

The implementation automatically downloads and processes:
- **WMT16 English-Romanian** dataset for translation
- **CC100** monolingual corpora for pre-training

### 3. Training

#### Option A: Direct Fine-tuning (Recommended)
```bash
# Fine-tune on English-Romanian translation
python scripts/train.py --mode finetune --batch_size 8 --max_epochs 3
```

#### Option B: Full Pipeline
```bash
# Step 1: Pre-training on monolingual data
python scripts/train.py --mode pretrain --batch_size 8 --max_steps 5000

# Step 2: Fine-tuning on translation data
python scripts/train.py --mode finetune --batch_size 8 --max_epochs 3
```

### 4. Evaluation

```bash
# Evaluate trained model
python scripts/train.py --mode evaluate --model_path checkpoints/translation/best_translation_model.pt
```

## Configuration

### Pre-training Configuration
Edit `configs/pretrain_config.json`:
```json
{
  "training": {
    "batch_size": 8,
    "max_steps": 5000,
    "learning_rate": 5e-4,
    "warmup_steps": 500
  },
  "data": {
    "max_length": 512,
    "mask_prob": 0.35,
    "poisson_lambda": 3.5
  }
}
```

### Translation Configuration
Edit `configs/translation_config.json`:
```json
{
  "training": {
    "batch_size": 8,
    "max_epochs": 3,
    "learning_rate": 3e-5,
    "warmup_steps": 500
  },
  "data": {
    "max_length": 128,
    "train_size": 10000
  }
}
```

## Advanced Usage

### Custom Training

```python
from src.model import MultilingualTranslationModel
from src.trainer import TranslationTrainer
from src.data import DataProcessor

# Initialize components
processor = DataProcessor()
train_loader, val_loader, test_loader = processor.create_dataloaders(batch_size=8)

model = MultilingualTranslationModel("facebook/mbart-large-50")

# Custom training
trainer = TranslationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=torch.device("cuda"),
    output_dir="custom_checkpoints"
)
trainer.train()
```

### Custom Evaluation

```python
from src.evaluation import TranslationEvaluator
from src.model import MultilingualTranslationModel

# Load model and evaluator
model = MultilingualTranslationModel("facebook/mbart-large-50")
evaluator = TranslationEvaluator(model, model.tokenizer, device)

# Evaluate on custom data
source_texts = ["Hello, how are you?", "I love machine learning."]
target_texts = ["Salut, ce mai faci?", "Îmi place învățarea automată."]

results, hypotheses, references = evaluator.evaluate_dataset(
    source_texts, target_texts, batch_size=2
)

print(f"BLEU Score: {results['bleu']:.4f}")
```

## Expected Results

### English-Romanian Translation Performance
Based on the original paper, expected results:

| Model | BLEU Score |
|-------|------------|
| Baseline (no pre-training) | ~20-25 |
| mBART (with pre-training) | ~30-35 |
| mBART + Fine-tuning | ~35-40 |

*Note: Results may vary based on training data size and hyperparameters*

## Training Time Estimates

| Hardware | Dataset Size | Training Time |
|----------|--------------|---------------|
| 1x RTX 3080 | 10K samples | ~2-3 hours |
| 4x V100 | 100K samples | ~4-6 hours |
| 8x A100 | Full dataset | ~2-3 days |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python scripts/train.py --batch_size 4 --max_length 64
   ```

2. **Dataset Download Issues**
   ```python
   # Use smaller fallback dataset
   processor = DataProcessor()
   data = processor.load_ted_talks("train")  # Instead of WMT
   ```

3. **Slow Training**
   ```bash
   # Use mixed precision
   pip install apex
   # Enable fp16 in configs
   ```

## File Structure

```
seq-2seq/
├── src/
│   ├── model.py              # Model implementations
│   ├── data.py               # Data processing
│   ├── trainer.py            # Training loops
│   └── evaluation.py         # Evaluation metrics
├── configs/
│   ├── pretrain_config.json  # Pre-training config
│   └── translation_config.json  # Translation config
├── scripts/
│   └── train.py              # Main training script
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
└── evaluation_results/       # Evaluation outputs
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{liu2020multilingual,
  title={Multilingual denoising pre-training for neural machine translation},
  author={Liu, Yinhan and Gu, Jiatao and Goyal, Naman and Li, Xian and Edunov, Sergey and Ghazvininejad, Marjan and Lewis, Mike and Zettlemoyer, Luke},
  journal={Transactions of the Association for Computational Linguistics},
  volume={8},
  pages={726--742},
  year={2020},
  publisher={MIT Press}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.