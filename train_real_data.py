#!/usr/bin/env python3
"""
Complete training script for English-Romanian mBART with real WMT16 data
"""

import torch
import os
import json
from src.data import DataProcessor
from src.model import MultilingualTranslationModel
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

def train_en_ro_real():
    """Train mBART on real English-Romanian data"""
    
    print("🚀 Training mBART on Real English-Romanian Data")
    print("=" * 60)
    
    # 1. Load real WMT16 data
    print("📊 Loading WMT16 English-Romanian dataset...")
    processor = DataProcessor()
    
    # Load small subsets for demonstration
    train_data = processor.load_wmt_en_ro('train')[:100]
    val_data = processor.load_wmt_en_ro('validation')[:20]
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    # Show sample data
    print("\n📋 Sample training data:")
    for i, item in enumerate(train_data[:3]):
        print(f"   {i+1}. EN: {item['source'][:60]}...")
        print(f"      RO: {item['target'][:60]}...")
    
    # 2. Initialize model and tokenizer
    print("\n🤖 Initializing mBART model...")
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device('cpu')  # Using CPU for this demo
    model.to(device)
    
    # 3. Create training loop
    print("\n🔧 Setting up training...")
    
    # Set language codes
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ro_RO"
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    # 4. Training function
    def train_epoch(model, data, tokenizer, optimizer, device):
        model.train()
        total_loss = 0
        
        for idx, item in enumerate(data):
            # Tokenize source and target
            source = item['source']
            target = item['target']
            
            # Prepare inputs
            inputs = tokenizer(
                source,
                text_target=target,
                max_length=64,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (idx + 1) % 10 == 0:
                print(f"   Step {idx+1}: Loss = {loss.item():.4f}")
        
        return total_loss / len(data)
    
    # 5. Train model
    print("\n🔥 Starting training...")
    print("   Model: mBART-large-50")
    print("   Data: WMT16 English-Romanian")
    print("   Device: CPU")
    print("   Epochs: 1")
    print("   Learning rate: 3e-5")
    print("   Max length: 64")
    
    avg_loss = train_epoch(model, train_data, tokenizer, optimizer, device)
    
    # 6. Save model
    os.makedirs('checkpoints/real_training', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/real_training/mbart_en_ro_real.pt')
    
    print(f"\n✅ Training complete! Average loss: {avg_loss:.4f}")
    
    # 7. Test model
    print("\n🎯 Testing model...")
    model.eval()
    
    test_sentences = [
        "Hello, how are you?",
        "I love machine learning.",
        "The weather is beautiful today.",
        "Thank you very much for your help."
    ]
    
    with torch.no_grad():
        for sentence in test_sentences:
            inputs = tokenizer(
                sentence,
                max_length=64,
                truncation=True,
                return_tensors='pt'
            )
            
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["ro_RO"],
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"EN: {sentence}")
            print(f"RO: {translation}")
            print()
    
    print("🎉 Training and evaluation complete!")
    print("\n💾 Model saved to: checkpoints/real_training/mbart_en_ro_real.pt")

if __name__ == "__main__":
    train_en_ro_real()