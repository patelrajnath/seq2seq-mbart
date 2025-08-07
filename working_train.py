#!/usr/bin/env python3
"""
Working training script with memory-efficient settings
"""

import sys
sys.path.insert(0, ".")

import torch
import argparse
from src.data import DataProcessor
from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator
from src.trainer import DenoisingPretrainingTrainer, TranslationTrainer
from transformers import MBartConfig

def run_efficient_pretraining(batch_size=2, max_steps=5):
    """Run pretraining with memory-efficient settings"""
    print("🚀 Efficient Pretraining")
    print("=" * 40)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Device: {device}")
        
        # Create processor
        processor = DataProcessor()
        vocab_size = len(processor.tokenizer)
        
        # Create small data
        train_loader = processor.create_pretrain_dataloaders(
            batch_size=batch_size,
            max_length=32,
            num_samples=max_steps * 2
        )
        val_loader = processor.create_pretrain_dataloaders(
            batch_size=batch_size,
            max_length=32,
            num_samples=10
        )
        
        # Create small model
        config = MBartConfig(
            vocab_size=vocab_size,
            d_model=256,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=512,
            decoder_ffn_dim=512,
            max_position_embeddings=32,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
        )
        
        model = MultilingualDenoisingPretraining(config)
        noise_generator = NoiseGenerator(processor.tokenizer)
        
        # Create trainer
        trainer = DenoisingPretrainingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir="checkpoints/pretrain_test",
            learning_rate=1e-4,
            warmup_steps=2,
            max_steps=max_steps
        )
        
        trainer.train()
        return True
        
    except Exception as e:
        print(f"❌ Pretraining failed: {e}")
        return False

def run_efficient_finetuning(batch_size=2, max_epochs=1):
    """Run finetuning with memory-efficient settings"""
    print("🚀 Efficient Finetuning")
    print("=" * 40)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Device: {device}")
        
        # Create processor
        processor = DataProcessor()
        
        # Create small data
        train_loader, val_loader, test_loader = processor.create_dataloaders(
            batch_size=batch_size,
            max_length=32,
            data_size=20
        )
        
        # Create model
        model = MultilingualTranslationModel("facebook/mbart-large-50-many-to-many-mmt")
        
        # Create trainer
        trainer = TranslationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir="checkpoints/translation_test",
            learning_rate=3e-5,
            warmup_steps=2,
            max_epochs=max_epochs
        )
        
        trainer.train()
        return True
        
    except Exception as e:
        print(f"❌ Finetuning failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=1)
    
    args = parser.parse_args()
    
    if args.mode == "pretrain":
        success = run_efficient_pretraining(args.batch_size, args.max_steps)
    else:
        success = run_efficient_finetuning(args.batch_size, args.max_epochs)
    
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n⚠️  Training completed with errors")

if __name__ == "__main__":
    main()