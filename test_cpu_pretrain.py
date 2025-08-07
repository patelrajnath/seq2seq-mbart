#!/usr/bin/env python3
"""
CPU-based pretraining test to verify indexing is fixed
"""

import sys
sys.path.insert(0, ".")

import torch
from src.data import DataProcessor, DenoisingPretrainDataset
from src.model import MultilingualDenoisingPretraining, NoiseGenerator
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def test_cpu_pretrain():
    """Test pretraining on CPU"""
    print("🧪 CPU Pretraining Test")
    print("=" * 40)
    
    try:
        device = torch.device("cpu")  # Force CPU
        print(f"📱 Device: {device}")
        
        # Create processor
        processor = DataProcessor()
        vocab_size = len(processor.tokenizer)
        print(f"Vocab size: {vocab_size}")
        
        # Create tiny dataset
        dataloader = processor.create_pretrain_dataloaders(
            batch_size=1,
            max_length=16,
            num_samples=5
        )
        
        # Create tiny model
        from transformers import MBartConfig
        config = MBartConfig(
            vocab_size=vocab_size,
            d_model=128,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
            max_position_embeddings=16,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
        )
        
        model = MultilingualDenoisingPretraining(config)
        model.to(device)
        noise_generator = NoiseGenerator(processor.tokenizer)
        
        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=1, num_training_steps=3
        )
        
        print("🎯 Starting 3 training steps...")
        model.train()
        
        for step, batch in enumerate(dataloader):
            if step >= 3:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Add noise
            batch["input_ids"] = noise_generator.add_noise(
                batch["input_ids"], noise_type="span_masking"
            )
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            print(f"Step {step+1}: loss={loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cpu_pretrain()
    if success:
        print("\n🎉 Pretraining indexing fixed!")
    else:
        print("\n⚠️  Still has issues")