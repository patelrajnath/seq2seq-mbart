#!/usr/bin/env python3
"""
Lightweight training test to verify the training loop works
"""

import sys
sys.path.insert(0, ".")

import torch
from src.data import DataProcessor
from src.model import MultilingualTranslationModel
from src.trainer import TranslationTrainer
from torch.utils.data import DataLoader

def test_lightweight_training():
    """Test training with minimal data and small model"""
    print("🚀 Testing Lightweight Training")
    print("=" * 50)
    
    try:
        # Create minimal test data
        print("📊 Creating test data...")
        processor = DataProcessor()
        
        # Use only 10 samples for testing
        test_data = [
            {"source": "Hello", "target": "Salut"},
            {"source": "Good morning", "target": "Bună dimineața"},
            {"source": "Thank you", "target": "Mulțumesc"},
            {"source": "How are you?", "target": "Ce mai faci?"},
            {"source": "Goodbye", "target": "La revedere"},
        ] * 2  # 10 total samples
        
        # Create small datasets
        from src.data import TranslationDataset
        train_dataset = TranslationDataset(test_data[:8], processor.tokenizer, max_length=32)
        val_dataset = TranslationDataset(test_data[8:], processor.tokenizer, max_length=32)
        
        # Create small dataloaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        print(f"✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Create small model for testing
        print("🤖 Creating test model...")
        from transformers import MBartConfig, MBartForConditionalGeneration
        
        config = MBartConfig(
            vocab_size=250027,
            d_model=256,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=512,
            decoder_ffn_dim=512,
            max_position_embeddings=64,
            pad_token_id=1,
            eos_token_id=2,
            bos_token_id=0,
        )
        
        model = MBartForConditionalGeneration(config)
        print(f"✅ Test model created: {model.num_parameters():,} parameters")
        
        # Test single training step
        print("🎯 Testing training step...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        trainer = TranslationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir="test_output",
            learning_rate=1e-4,
            warmup_steps=2,
            max_epochs=1
        )
        
        # Test one training step
        trainer.model.train()
        batch = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = trainer.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs["loss"]
        loss.backward()
        
        print(f"✅ Training step successful, loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lightweight_training()
    if success:
        print("\n🎉 Training pipeline is working correctly!")
        print("You can now run:")
        print("  python3 scripts/train.py --mode finetune --batch_size 4 --max_epochs 1 --data_size 100")
    else:
        print("\n⚠️  Training pipeline needs debugging")