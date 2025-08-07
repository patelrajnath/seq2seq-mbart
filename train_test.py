#!/usr/bin/env python3
"""
Complete training test with proper error handling and small dataset
"""

import sys
sys.path.insert(0, ".")

import torch
from src.data import DataProcessor
from src.model import MultilingualTranslationModel
from src.trainer import TranslationTrainer
from torch.utils.data import DataLoader

def run_small_training_test():
    """Run training with minimal data to verify everything works"""
    print("🚀 Small Training Test")
    print("=" * 50)
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Device: {device}")
        
        # Create processor
        processor = DataProcessor()
        print("✅ Data processor initialized")
        
        # Create small dataset
        small_data = [
            {"source": "Hello", "target": "Salut"},
            {"source": "Good morning", "target": "Bună dimineața"},
            {"source": "Thank you", "target": "Mulțumesc"},
            {"source": "How are you?", "target": "Ce mai faci?"},
            {"source": "Goodbye", "target": "La revedere"},
            {"source": "Please", "target": "Vă rog"},
            {"source": "Welcome", "target": "Bine ați venit"},
            {"source": "Machine learning", "target": "Învățare automată"},
        ]
        
        # Split into train/val
        train_data = small_data[:6]
        val_data = small_data[6:]
        
        # Create datasets
        from src.data import TranslationDataset
        train_dataset = TranslationDataset(train_data, processor.tokenizer, max_length=32)
        val_dataset = TranslationDataset(val_data, processor.tokenizer, max_length=32)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        print(f"✅ Data: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Create model
        model = MultilingualTranslationModel("facebook/mbart-large-50-many-to-many-mmt")
        print(f"✅ Model: {model.model.num_parameters():,} parameters")
        
        # Create trainer
        trainer = TranslationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir="test_checkpoints",
            learning_rate=3e-5,
            warmup_steps=2,
            max_epochs=1
        )
        
        # Run one epoch
        print("🎯 Starting training...")
        trainer.train()
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_small_training_test()
    if success:
        print("\n🎉 Training completed successfully!")
        print("\nNow you can run:")
        print("  python3 scripts/train.py --mode finetune --batch_size 4 --max_epochs 1")
    else:
        print("\n⚠️  Training needs more debugging")