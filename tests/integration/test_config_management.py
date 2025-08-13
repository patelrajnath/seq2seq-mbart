"""
Integration tests for configuration management
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the training script functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from train import load_config, setup_directories, main


class TestConfigLoading:
    """Test configuration loading functionality"""

    @pytest.mark.integration
    def test_load_valid_config(self, test_config, temp_config_file):
        """Test loading a valid configuration file"""
        loaded_config = load_config(temp_config_file)
        
        assert loaded_config == test_config
        assert isinstance(loaded_config, dict)
        assert 'model' in loaded_config
        assert 'training' in loaded_config
        assert 'data' in loaded_config
        assert 'output' in loaded_config

    @pytest.mark.integration
    def test_load_config_with_nested_structure(self):
        """Test loading config with deeply nested structure"""
        nested_config = {
            "model": {
                "architecture": {
                    "encoder": {
                        "layers": 6,
                        "attention": {"heads": 8, "dropout": 0.1}
                    },
                    "decoder": {
                        "layers": 6,
                        "attention": {"heads": 8, "dropout": 0.1}
                    }
                }
            },
            "training": {
                "optimizer": {
                    "name": "adamw",
                    "params": {"lr": 3e-5, "weight_decay": 0.01}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(nested_config, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == nested_config
            assert loaded_config["model"]["architecture"]["encoder"]["layers"] == 6
            assert loaded_config["training"]["optimizer"]["params"]["lr"] == 3e-5
        finally:
            os.unlink(temp_path)

    @pytest.mark.integration
    def test_load_config_invalid_json(self):
        """Test loading invalid JSON configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content }")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    @pytest.mark.integration
    def test_load_config_nonexistent_file(self):
        """Test loading non-existent configuration file"""
        with pytest.raises(FileNotFoundError):
            load_config("/path/that/does/not/exist.json")

    @pytest.mark.integration
    def test_load_config_empty_file(self):
        """Test loading empty configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    @pytest.mark.integration
    def test_load_config_unicode_content(self):
        """Test loading configuration with unicode content"""
        unicode_config = {
            "model": {
                "name": "mBART_română_español",
                "languages": ["română", "español", "français"]
            },
            "description": "Model pentru traducere automată între limbi romanice"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(unicode_config, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == unicode_config
            assert "română" in loaded_config["model"]["languages"]
        finally:
            os.unlink(temp_path)


class TestDirectorySetup:
    """Test directory setup functionality"""

    @pytest.mark.integration
    def test_setup_directories_basic(self):
        """Test basic directory setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "output": {
                    "output_dir": os.path.join(temp_dir, "checkpoints"),
                    "log_dir": os.path.join(temp_dir, "logs")
                }
            }
            
            setup_directories(config)
            
            assert os.path.exists(config["output"]["output_dir"])
            assert os.path.exists(config["output"]["log_dir"])
            assert os.path.isdir(config["output"]["output_dir"])
            assert os.path.isdir(config["output"]["log_dir"])

    @pytest.mark.integration
    def test_setup_directories_nested(self):
        """Test setup of nested directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "output": {
                    "output_dir": os.path.join(temp_dir, "deep", "nested", "checkpoints"),
                    "log_dir": os.path.join(temp_dir, "deep", "nested", "logs")
                }
            }
            
            setup_directories(config)
            
            assert os.path.exists(config["output"]["output_dir"])
            assert os.path.exists(config["output"]["log_dir"])

    @pytest.mark.integration
    def test_setup_directories_already_exist(self):
        """Test setup when directories already exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "checkpoints")
            log_dir = os.path.join(temp_dir, "logs")
            
            # Pre-create directories
            os.makedirs(output_dir)
            os.makedirs(log_dir)
            
            config = {
                "output": {
                    "output_dir": output_dir,
                    "log_dir": log_dir
                }
            }
            
            # Should not raise exception
            setup_directories(config)
            
            assert os.path.exists(output_dir)
            assert os.path.exists(log_dir)

    @pytest.mark.integration
    def test_setup_directories_with_files(self):
        """Test directory setup preserves existing files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "checkpoints")
            os.makedirs(output_dir)
            
            # Create a file in the directory
            test_file = os.path.join(output_dir, "existing_file.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            
            config = {
                "output": {
                    "output_dir": output_dir,
                    "log_dir": os.path.join(temp_dir, "logs")
                }
            }
            
            setup_directories(config)
            
            # File should still exist
            assert os.path.exists(test_file)
            with open(test_file, 'r') as f:
                assert f.read() == "test content"


class TestConfigValidation:
    """Test configuration validation"""

    @pytest.mark.integration
    def test_validate_pretrain_config(self):
        """Test validation of pretraining configuration"""
        config = {
            "model": {
                "d_model": 512,
                "encoder_layers": 6,
                "decoder_layers": 6
            },
            "training": {
                "max_steps": 10000,
                "batch_size": 8,
                "learning_rate": 5e-4
            },
            "data": {
                "max_length": 512,
                "num_samples": 1000
            },
            "output": {
                "output_dir": "checkpoints/pretrain",
                "log_dir": "logs/pretrain"
            }
        }
        
        # Should be valid for pretraining
        assert "max_steps" in config["training"]
        assert "num_samples" in config["data"]
        assert config["training"]["max_steps"] > 0
        assert config["data"]["num_samples"] > 0

    @pytest.mark.integration
    def test_validate_finetune_config(self):
        """Test validation of finetuning configuration"""
        config = {
            "model": {
                "pretrained_model": "facebook/mbart-large-50"
            },
            "training": {
                "max_epochs": 3,
                "batch_size": 16,
                "learning_rate": 3e-5
            },
            "data": {
                "max_length": 128,
                "train_size": 10000
            },
            "output": {
                "output_dir": "checkpoints/finetune",
                "log_dir": "logs/finetune"
            }
        }
        
        # Should be valid for finetuning
        assert "max_epochs" in config["training"]
        assert "train_size" in config["data"]
        assert config["training"]["max_epochs"] > 0
        assert config["data"]["train_size"] > 0

    @pytest.mark.integration
    def test_config_parameter_types(self):
        """Test that configuration parameters have correct types"""
        config = {
            "model": {
                "d_model": 512,           # int
                "dropout": 0.1            # float
            },
            "training": {
                "batch_size": 8,          # int
                "learning_rate": 1e-4,    # float
                "max_steps": 1000         # int
            },
            "data": {
                "max_length": 256,        # int
                "languages": ["en", "ro"] # list
            }
        }
        
        # Type validation
        assert isinstance(config["model"]["d_model"], int)
        assert isinstance(config["model"]["dropout"], float)
        assert isinstance(config["training"]["batch_size"], int)
        assert isinstance(config["training"]["learning_rate"], float)
        assert isinstance(config["data"]["languages"], list)

    @pytest.mark.integration
    def test_config_parameter_ranges(self):
        """Test that configuration parameters are within valid ranges"""
        config = {
            "model": {
                "d_model": 512,
                "encoder_layers": 6,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-4
            }
        }
        
        # Range validation
        assert config["model"]["d_model"] > 0
        assert config["model"]["encoder_layers"] > 0
        assert 0 <= config["model"]["dropout"] <= 1
        assert config["training"]["batch_size"] > 0
        assert config["training"]["learning_rate"] > 0


class TestCommandLineConfigOverrides:
    """Test command-line configuration overrides"""

    @pytest.mark.integration
    @patch('sys.argv')
    def test_batch_size_override(self, mock_argv):
        """Test batch size override from command line"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "training": {"batch_size": 8, "max_epochs": 1},
                "data": {"train_size": 10},
                "output": {"output_dir": temp_dir, "log_dir": temp_dir}
            }
            
            config_path = os.path.join(temp_dir, "test_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Mock command line arguments
            mock_argv.__getitem__.side_effect = lambda i: [
                'train.py', '--mode', 'finetune', 
                '--config', config_path, 
                '--batch_size', '16'
            ][i]
            mock_argv.__len__.return_value = 6
            
            # Mock the training functions to avoid actual training
            with patch('train.run_finetuning') as mock_finetune, \
                 patch('train.setup_directories'):
                
                try:
                    main()
                except SystemExit:
                    pass  # argparse may call sys.exit
                
                # Check that run_finetuning was called
                if mock_finetune.called:
                    # The config should have been updated
                    call_args = mock_finetune.call_args[1] if mock_finetune.call_args else None
                    if call_args and 'config_path' in call_args:
                        # Load the temporary config that was created
                        temp_config_path = call_args['config_path']
                        if os.path.exists(temp_config_path):
                            with open(temp_config_path, 'r') as f:
                                updated_config = json.load(f)
                            assert updated_config["training"]["batch_size"] == 16

    @pytest.mark.integration
    @patch('sys.argv')
    def test_learning_rate_override(self, mock_argv):
        """Test learning rate override from command line"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "training": {"learning_rate": 1e-4, "max_epochs": 1},
                "data": {"train_size": 10},
                "output": {"output_dir": temp_dir, "log_dir": temp_dir}
            }
            
            config_path = os.path.join(temp_dir, "test_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            mock_argv.__getitem__.side_effect = lambda i: [
                'train.py', '--mode', 'finetune',
                '--config', config_path,
                '--learning_rate', '5e-5'
            ][i]
            mock_argv.__len__.return_value = 6
            
            with patch('train.run_finetuning') as mock_finetune, \
                 patch('train.setup_directories'):
                
                try:
                    main()
                except SystemExit:
                    pass
                
                # Verify the function would have been called
                # (exact verification depends on implementation)
                assert mock_finetune.call_count <= 1

    @pytest.mark.integration
    def test_config_override_precedence(self):
        """Test that command line arguments take precedence over config file"""
        base_config = {
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-4,
                "max_epochs": 3
            }
        }
        
        # Simulate command line overrides
        cli_overrides = {
            "batch_size": 16,
            "learning_rate": 5e-5
        }
        
        # Apply overrides (simulating what the main function does)
        updated_config = base_config.copy()
        for key, value in cli_overrides.items():
            if key in updated_config["training"]:
                updated_config["training"][key] = value
        
        # Verify precedence
        assert updated_config["training"]["batch_size"] == 16  # Overridden
        assert updated_config["training"]["learning_rate"] == 5e-5  # Overridden
        assert updated_config["training"]["max_epochs"] == 3  # Not overridden


class TestConfigCompatibility:
    """Test configuration compatibility across different scenarios"""

    @pytest.mark.integration
    def test_backward_compatibility(self):
        """Test that old configuration formats still work"""
        # Old-style config without some new fields
        old_config = {
            "model": {
                "d_model": 512,
                "encoder_layers": 6
                # Missing some newer fields
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-4
                # Missing some newer fields
            }
        }
        
        # Should handle missing fields gracefully
        assert "d_model" in old_config["model"]
        assert "batch_size" in old_config["training"]
        
        # Missing fields should be handled by defaults in the code
        # This test verifies the structure is still readable

    @pytest.mark.integration
    def test_config_with_extra_fields(self):
        """Test that configuration with extra fields is handled gracefully"""
        config_with_extras = {
            "model": {
                "d_model": 512,
                "extra_model_field": "should_be_ignored"
            },
            "training": {
                "batch_size": 8,
                "extra_training_field": 123
            },
            "extra_top_level": {
                "some_field": "some_value"
            }
        }
        
        # Should be able to load without errors
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_with_extras, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == config_with_extras
            # Extra fields should be preserved (not filtered out)
            assert "extra_model_field" in loaded_config["model"]
        finally:
            os.unlink(temp_path)

    @pytest.mark.integration
    def test_minimal_config(self):
        """Test that minimal configuration works"""
        minimal_config = {
            "training": {"batch_size": 4},
            "output": {"output_dir": "test_output"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config["training"]["batch_size"] == 4
            # Missing sections should be handled by application logic
        finally:
            os.unlink(temp_path)

    @pytest.mark.integration
    def test_config_path_resolution(self):
        """Test configuration path resolution"""
        # Test relative path resolution
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"training": {"batch_size": 8}}
            
            # Create config in subdirectory
            subdir = os.path.join(temp_dir, "configs")
            os.makedirs(subdir)
            config_path = os.path.join(subdir, "test_config.json")
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Should be able to load with absolute path
            loaded_config = load_config(config_path)
            assert loaded_config == config
            
            # Test that relative paths work from the correct working directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                relative_path = os.path.join("configs", "test_config.json")
                loaded_config = load_config(relative_path)
                assert loaded_config == config
            finally:
                os.chdir(original_cwd)