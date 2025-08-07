#!/usr/bin/env python3
"""
Debug embedding layer indexing issues
"""

import sys
sys.path.insert(0, ".")

import torch
from src.data import DataProcessor
from transformers import MBartConfig, MBartForConditionalGeneration

def debug_embedding_issue():
    """Debug the embedding indexing issue"""
    print("🔍 Debugging Embedding Layer Issue")
    print("=" * 50)
    
    try:
        # Create processor and get tokenizer
        processor = DataProcessor()
        tokenizer = processor.tokenizer
        
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Tokenizer special tokens:")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")
        print(f"  eos_token_id: {tokenizer.eos_token_id}")
        print(f"  bos_token_id: {tokenizer.bos_token_id}")
        print(f"  mask_token_id: {tokenizer.mask_token_id}")
        
        # Test with actual tokenizer
        test_text = "Hello world"
        encoded = tokenizer(test_text, max_length=10, truncation=True, return_tensors="pt")
        print(f"\nTest encoding:")
        print(f"Input IDs: {encoded['input_ids']}")
        print(f"Max token ID: {encoded['input_ids'].max()}")
        print(f"Min token ID: {encoded['input_ids'].min()}")
        
        # Create model with proper vocab size
        vocab_size = len(tokenizer)
        print(f"\nUsing vocab size: {vocab_size}")
        
        config = MBartConfig(
            vocab_size=vocab_size,  # Use actual tokenizer vocab size
            d_model=256,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=512,
            decoder_ffn_dim=512,
            max_position_embeddings=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        
        model = MBartForConditionalGeneration(config)
        print(f"✅ Model created successfully")
        print(f"Model vocab size: {model.config.vocab_size}")
        
        # Test with dummy data
        dummy_input = torch.randint(0, vocab_size-1, (2, 10))
        dummy_labels = torch.randint(0, vocab_size-1, (2, 10))
        
        print(f"\nTesting with dummy data:")
        print(f"Input range: {dummy_input.min()}-{dummy_input.max()}")
        
        with torch.no_grad():
            outputs = model(input_ids=dummy_input, labels=dummy_labels)
            print(f"✅ Forward pass successful: loss={outputs.loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_embedding_issue()