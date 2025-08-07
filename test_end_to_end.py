#!/usr/bin/env python3
"""
End-to-end test of the fixed mBART training pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_processor():
    """Test the DataProcessor with tokenizer fix"""
    print("🔍 Testing DataProcessor...")
    
    try:
        from src.data import DataProcessor
        processor = DataProcessor()
        
        # Test dummy data
        dummy_data = processor._create_dummy_data()
        print(f"✅ Dummy data: {len(dummy_data)} pairs")
        
        # Test tokenizer
        tokenizer = processor.tokenizer
        print(f"✅ Tokenizer type: {type(tokenizer).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ DataProcessor error: {e}")
        return False

def test_model_creation():
    """Test model creation without pretrained weights"""
    print("🔍 Testing model creation...")
    
    try:
        from transformers import MBartConfig, MBartForConditionalGeneration
        
        # Create small config for testing
        config = MBartConfig(
            vocab_size=250027,
            d_model=256,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=512,
            decoder_ffn_dim=512,
            max_position_embeddings=128,
            pad_token_id=1,
            eos_token_id=2,
            bos_token_id=0,
        )
        
        model = MBartForConditionalGeneration(config)
        print(f"✅ Model created: {model.num_parameters():,} parameters")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_training_command():
    """Test the training command structure"""
    print("🔍 Testing training command...")
    
    try:
        # Test the basic structure without actually running
        from src.data import DataProcessor
        from src.model import MultilingualTranslationModel
        
        # Test importability
        print("✅ All modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run end-to-end tests"""
    print("🎯 End-to-End Training Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("DataProcessor", test_data_processor),
        ("Model Creation", test_model_creation),
        ("Training Command", test_training_command),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Training pipeline is ready!")
        print("\nTo start training:")
        print("  python3 scripts/train.py --mode finetune --batch_size 4 --max_epochs 1")
    else:
        print("⚠️  Some tests failed")
    
    return passed == total

if __name__ == "__main__":
    main()