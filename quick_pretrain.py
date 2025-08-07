#!/usr/bin/env python3
"""
Quick pretraining test with minimal memory usage
"""

import sys
sys.path.insert(0, ".")

import torch
from src.data import DataProcessor
from torch.utils.data import DataLoader
from transformers import MBartConfig, MBartForConditionalGeneration
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def run_mini_pretrain():
    """Run minimal pretraining test"""
    print("🚀 Mini Pretraining Test")
    print("=" * 40)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Device: {device}")
        
        # Create small data
        processor = DataProcessor()
        dataloader = processor.create_pretrain_dataloaders(
            batch_size=1,
            max_length=32,
            num_samples=10
        )
        
        print(f"✅ Data loaded: {len(dataloader)} batches")
        
        # Create tiny model
        config = MBartConfig(
            vocab_size=250027,
            d_model=128,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
            max_position_embeddings=32,
        )
        
        model = MBartForConditionalGeneration(config)
        print(f"✅ Tiny model: {model.num_parameters():,} parameters")
        
        # Simple training loop
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=2, num_training_steps=5
        )
        
        model.to(device)
        model.train()
        
        print("🎯 Training 5 steps...")
        for step, batch in enumerate(dataloader):
            if step >= 5:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            print(f"Step {step+1}: loss={loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = run_mini_pretrain()
    if success:
        print("\n🎉 Mini pretraining completed!")
    else:
        print("\n⚠️  Pretraining failed")