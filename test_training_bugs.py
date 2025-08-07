#!/usr/bin/env python3
"""
Comprehensive test script to identify bugs in mBART training pipeline
"""

import torch
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_pipeline():
    """Test data loading and preprocessing"""
    print("🔍 Testing data pipeline...")
    
    try:
        from src.data import DataProcessor
        processor = DataProcessor()
        
        # Test dummy data creation
        dummy_data = processor._create_dummy_data()
        print(f"   ✅ Dummy data created: {len(dummy_data)} pairs")
        
        # Test data format
        for i, item in enumerate(dummy_data[:3]):
            print(f"   Sample {i+1}: {item}")
            assert "source" in item and "target" in item, "Missing source/target keys"
            assert isinstance(item["source"], str), "Source not string"
            assert isinstance(item["target"], str), "Target not string"
        
        return True
    except Exception as e:
        print(f"   ❌ Data pipeline error: {e}")
        return False

def test_model_initialization():
    """Test model initialization without loading pretrained weights"""
    print("🔍 Testing model initialization...")
    
    try:
        from src.model import MultilingualTranslationModel
        
        # Test model creation without pretrained weights
        print("   Testing model config...")
        from transformers import MBartConfig, MBartForConditionalGeneration
        
        config = MBartConfig(
            vocab_size=250027,
            d_model=512,  # Smaller for testing
            encoder_layers=2,  # Smaller for testing
            decoder_layers=2,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_ffn_dim=1024,
            decoder_ffn_dim=1024,
            max_position_embeddings=512,
        )
        
        model = MBartForConditionalGeneration(config)
        print(f"   ✅ Model initialized: {model.num_parameters()} parameters")
        
        return True
    except Exception as e:
        print(f"   ❌ Model initialization error: {e}")
        return False

def test_training_loop():
    """Test training loop with dummy tensors"""
    print("🔍 Testing training loop...")
    
    try:
        from transformers import MBartConfig, MBartForConditionalGeneration
        
        # Create small model
        config = MBartConfig(
            vocab_size=1000,
            d_model=128,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
            max_position_embeddings=64,
            pad_token_id=1,
            eos_token_id=2,
            bos_token_id=0,
        )
        
        model = MBartForConditionalGeneration(config)
        
        # Create dummy batch
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.randint(3, 50, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(3, 50, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        print(f"   ✅ Forward pass successful, loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        print(f"   ✅ Backward pass successful")
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"   ⚠️  No gradient for {name}")
            elif torch.isnan(param.grad).any():
                print(f"   ❌ NaN gradient in {name}")
                return False
        
        print("   ✅ Gradient check passed")
        return True
    except Exception as e:
        print(f"   ❌ Training loop error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_files():
    """Test configuration files for consistency"""
    print("🔍 Testing configuration files...")
    
    try:
        with open("configs/translation_config.json", "r") as f:
            trans_config = json.load(f)
        
        with open("configs/pretrain_config.json", "r") as f:
            pretrain_config = json.load(f)
        
        # Check required keys
        required_keys = ["training", "data", "output"]
        for config_name, config in [("translation", trans_config), ("pretrain", pretrain_config)]:
            for key in required_keys:
                if key not in config:
                    print(f"   ❌ Missing {key} in {config_name} config")
                    return False
        
        # Check batch sizes
        trans_batch = trans_config["training"]["batch_size"]
        pretrain_batch = pretrain_config["training"]["batch_size"]
        
        print(f"   Translation batch_size: {trans_batch}")
        print(f"   Pre-training batch_size: {pretrain_batch}")
        
        # Check learning rates
        trans_lr = trans_config["training"]["learning_rate"]
        pretrain_lr = pretrain_config["training"]["learning_rate"]
        
        print(f"   Translation learning_rate: {trans_lr}")
        print(f"   Pre-training learning_rate: {pretrain_lr}")
        
        return True
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

def test_memory_requirements():
    """Test memory requirements estimation"""
    print("🔍 Testing memory requirements...")
    
    try:
        from transformers import MBartConfig, MBartForConditionalGeneration
        
        # Test with different model sizes
        configs = [
            ("small", {"d_model": 256, "encoder_layers": 2, "decoder_layers": 2}),
            ("medium", {"d_model": 512, "encoder_layers": 6, "decoder_layers": 6}),
            ("large", {"d_model": 1024, "encoder_layers": 12, "decoder_layers": 12}),
        ]
        
        for name, params in configs:
            config = MBartConfig(
                vocab_size=250027,
                d_model=params["d_model"],
                encoder_layers=params["encoder_layers"],
                decoder_layers=params["decoder_layers"],
                encoder_attention_heads=16,
                decoder_attention_heads=16,
                encoder_ffn_dim=params["d_model"] * 4,
                decoder_ffn_dim=params["d_model"] * 4,
                max_position_embeddings=1024,
            )
            
            model = MBartForConditionalGeneration(config)
            param_count = model.num_parameters()
            estimated_memory = param_count * 4 / (1024**3)  # GB for float32
            
            print(f"   {name}: {param_count:,} params ~ {estimated_memory:.2f}GB")
        
        return True
    except Exception as e:
        print(f"   ❌ Memory test error: {e}")
        return False

def test_device_compatibility():
    """Test device compatibility"""
    print("🔍 Testing device compatibility...")
    
    try:
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   ⚠️  CUDA not available, will use CPU")
        
        # Test device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        return True
    except Exception as e:
        print(f"   ❌ Device test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 mBART Training Bug Detection Report")
    print("=" * 60)
    
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Model Initialization", test_model_initialization),
        ("Training Loop", test_training_loop),
        ("Configuration Files", test_configuration_files),
        ("Memory Requirements", test_memory_requirements),
        ("Device Compatibility", test_device_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY REPORT")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Training should work correctly.")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()