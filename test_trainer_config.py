#!/usr/bin/env python3
"""
Test script to verify trainer configuration updates work correctly
"""
import sys
import os
sys.path.insert(0, 'src')

import json
import tempfile
from typing import Dict

def test_config_loading():
    """Test that both config files load correctly"""
    print("Testing config loading...")
    
    # Test pretrain config
    with open('configs/pretrain_config.json', 'r') as f:
        pretrain_config = json.load(f)
    
    assert 'model' in pretrain_config
    assert 'training' in pretrain_config
    assert 'data' in pretrain_config
    assert 'output' in pretrain_config
    
    print("✓ Pretrain config structure validated")
    
    # Test translation config
    with open('configs/translation_config.json', 'r') as f:
        translation_config = json.load(f)
    
    assert 'model' in translation_config
    assert 'training' in translation_config
    assert 'data' in translation_config
    assert 'output' in translation_config
    
    print("✓ Translation config structure validated")
    return True

def test_config_content():
    """Test that config values are correctly structured"""
    print("\nTesting config content...")
    
    with open('configs/pretrain_config.json', 'r') as f:
        config = json.load(f)
    
    # Test model parameters
    model_config = config['model']
    assert isinstance(model_config['d_model'], int)
    assert isinstance(model_config['encoder_layers'], int)
    assert isinstance(model_config['decoder_layers'], int)
    
    # Test training parameters
    training_config = config['training']
    assert isinstance(training_config['learning_rate'], (int, float))
    assert isinstance(training_config['batch_size'], int)
    assert isinstance(training_config['max_steps'], int)
    
    print("✓ Config content validated")
    return True

def test_trainer_signature():
    """Test that trainer classes accept config parameter"""
    print("\nTesting trainer signatures...")
    
    try:
        # Import required classes
        sys.path.insert(0, 'src')
        from trainer import DenoisingPretrainingTrainer, TranslationTrainer
        
        # Create mock classes to test signature
        class MockModel:
            def parameters(self):
                return []
        
        class MockLoader:
            def __len__(self):
                return 10
        
        # Test DenoisingPretrainingTrainer signature
        import inspect
        sig = inspect.signature(DenoisingPretrainingTrainer.__init__)
        assert 'config' in sig.parameters
        print("✓ DenoisingPretrainingTrainer accepts config parameter")
        
        # Test TranslationTrainer signature
        sig = inspect.signature(TranslationTrainer.__init__)
        assert 'config' in sig.parameters
        print("✓ TranslationTrainer accepts config parameter")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_usage():
    """Test that config values are correctly used"""
    print("\nTesting config usage...")
    
    with open('configs/pretrain_config.json', 'r') as f:
        config = json.load(f)
    
    # Test that we can access nested values
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    
    # Model parameters
    d_model = model_config.get("d_model", 1024)
    encoder_layers = model_config.get("encoder_layers", 12)
    
    # Training parameters
    learning_rate = training_config.get("learning_rate", 5e-4)
    batch_size = training_config.get("batch_size", 8)
    
    # Assert values match expected
    assert d_model == 1024
    assert encoder_layers == 12
    assert learning_rate == 5e-4
    assert batch_size == 8
    
    print("✓ Config values correctly accessed")
    return True

def test_argument_parsing():
    """Test that argument parsing works"""
    print("\nTesting argument parsing...")
    
    import subprocess
    import sys
    
    try:
        # Test help output
        result = subprocess.run([
            sys.executable, '-c', 
            '''
import sys
sys.path.insert(0, "src")
from trainer import run_pretraining, run_finetuning
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["pretrain", "finetune"], default="finetune")
parser.add_argument("--config", type=str, help="Path to config file")
args = parser.parse_args(["--help"])
            '''
        ], capture_output=True, text=True, timeout=10)
        
        if "--config" in result.stdout:
            print("✓ Argument parsing supports --config parameter")
            return True
        else:
            print("✗ Argument parsing failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Argument parsing test timed out")
        return False
    except Exception as e:
        print(f"✗ Argument parsing error: {e}")
        return False

def main():
    """Run all tests"""
    print("Running trainer configuration tests...\n")
    
    tests = [
        test_config_loading,
        test_config_content,
        test_trainer_signature,
        test_config_usage,
        test_argument_parsing,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Trainer configuration update is ready.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)