"""
Integration tests for the complete training pipeline
"""

import pytest
import torch
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.trainer import run_pretraining, run_finetuning
from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel
from src.data import DataProcessor
from src.evaluation import evaluate_model


class TestPretrainingPipeline:
    """Test pretraining pipeline integration"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pretraining_pipeline_minimal(self, test_config, temp_checkpoint_dir):
        """Test minimal pretraining pipeline"""
        # Modify config for very quick training
        test_config['training']['max_steps'] = 2
        test_config['training']['eval_interval'] = 1
        test_config['training']['save_interval'] = 1
        test_config['data']['num_samples'] = 5
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        # Create temporary config file
        config_path = os.path.join(temp_checkpoint_dir, "test_pretrain_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Mock wandb to avoid issues
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            
            # Run pretraining
            run_pretraining(config_path)
        
        # Check that checkpoints were created
        checkpoint_files = [f for f in os.listdir(temp_checkpoint_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) > 0
        
        # Check that at least one checkpoint can be loaded
        checkpoint_path = os.path.join(temp_checkpoint_dir, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'step', 'epoch']
        for key in required_keys:
            assert key in checkpoint

    @pytest.mark.integration
    def test_pretraining_with_custom_model_config(self, test_config, temp_checkpoint_dir):
        """Test pretraining with custom model configuration"""
        # Modify model config for testing
        test_config['model']['d_model'] = 64
        test_config['model']['encoder_layers'] = 1
        test_config['model']['decoder_layers'] = 1
        test_config['training']['max_steps'] = 1
        test_config['data']['num_samples'] = 3
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "custom_model_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            
            # Should not raise any exceptions
            run_pretraining(config_path)

    @pytest.mark.integration
    def test_pretraining_config_override(self, test_config, temp_checkpoint_dir):
        """Test that config values are properly used in pretraining"""
        custom_lr = 1e-3
        custom_batch_size = 4
        
        test_config['training']['learning_rate'] = custom_lr
        test_config['training']['batch_size'] = custom_batch_size
        test_config['training']['max_steps'] = 1
        test_config['data']['num_samples'] = 5
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "override_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb, \
             patch('src.trainer.DataProcessor') as mock_processor_class:
            
            mock_wandb.init.return_value = MagicMock()
            
            # Mock data processor to capture batch size
            mock_processor = MagicMock()
            mock_processor.create_pretrain_dataloaders.return_value = MagicMock()
            mock_processor.tokenizer = MagicMock()
            mock_processor.tokenizer.__len__ = MagicMock(return_value=1000)
            mock_processor.tokenizer.pad_token_id = 1
            mock_processor.tokenizer.eos_token_id = 2
            mock_processor.tokenizer.bos_token_id = 0
            mock_processor_class.return_value = mock_processor
            
            run_pretraining(config_path)
            
            # Verify batch_size was passed correctly
            call_args = mock_processor.create_pretrain_dataloaders.call_args
            assert call_args.kwargs['batch_size'] == custom_batch_size


class TestFinetuningPipeline:
    """Test finetuning pipeline integration"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_finetuning_pipeline_minimal(self, test_config, temp_checkpoint_dir):
        """Test minimal finetuning pipeline"""
        # Modify config for very quick training
        test_config['training']['max_epochs'] = 1
        test_config['training']['batch_size'] = 2
        test_config['data']['train_size'] = 5
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "test_finetune_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            
            # Run finetuning
            run_finetuning(config_path)
        
        # Check that checkpoints were created
        checkpoint_files = [f for f in os.listdir(temp_checkpoint_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) > 0

    @pytest.mark.integration
    @patch('src.trainer.MultilingualTranslationModel')
    def test_finetuning_with_pretrained_model(self, mock_model_class, test_config, temp_checkpoint_dir):
        """Test finetuning with pretrained model loading"""
        # Configure for minimal training
        test_config['training']['max_epochs'] = 1
        test_config['data']['train_size'] = 3
        test_config['model']['pretrained_model'] = "facebook/mbart-large-50"
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "pretrained_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Mock model to avoid downloading
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        mock_model_class.return_value = mock_model
        
        with patch('src.trainer.wandb') as mock_wandb, \
             patch('src.trainer.DataProcessor') as mock_processor_class:
            
            mock_wandb.init.return_value = MagicMock()
            
            # Mock data processor
            mock_processor = MagicMock()
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_test_loader = MagicMock()
            
            # Mock dataloader lengths and iteration
            mock_train_loader.__len__ = MagicMock(return_value=2)
            mock_train_loader.__iter__ = MagicMock(return_value=iter([
                {'input_ids': torch.randint(0, 1000, (2, 10)),
                 'attention_mask': torch.ones(2, 10),
                 'labels': torch.randint(0, 1000, (2, 10))}
            ]))
            
            mock_val_loader.__iter__ = MagicMock(return_value=iter([
                {'input_ids': torch.randint(0, 1000, (2, 10)),
                 'attention_mask': torch.ones(2, 10),
                 'labels': torch.randint(0, 1000, (2, 10))}
            ]))
            
            mock_processor.create_dataloaders.return_value = (
                mock_train_loader, mock_val_loader, mock_test_loader
            )
            mock_processor_class.return_value = mock_processor
            
            run_finetuning(config_path)
            
            # Verify pretrained model was loaded with correct name
            mock_model_class.assert_called_once_with("facebook/mbart-large-50")

    @pytest.mark.integration
    def test_finetuning_data_size_parameter(self, test_config, temp_checkpoint_dir):
        """Test that data_size parameter is properly handled in finetuning"""
        custom_data_size = 10
        
        test_config['training']['max_epochs'] = 1
        test_config['data']['train_size'] = custom_data_size
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "data_size_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb, \
             patch('src.trainer.DataProcessor') as mock_processor_class:
            
            mock_wandb.init.return_value = MagicMock()
            
            mock_processor = MagicMock()
            mock_processor.create_dataloaders.return_value = (
                MagicMock(), MagicMock(), MagicMock()
            )
            mock_processor_class.return_value = mock_processor
            
            run_finetuning(config_path)
            
            # Verify data_size was passed correctly
            call_args = mock_processor.create_dataloaders.call_args
            assert call_args.kwargs['data_size'] == custom_data_size


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pretrain_to_finetune_pipeline(self, test_config, temp_checkpoint_dir):
        """Test pretraining followed by finetuning"""
        # Configure for minimal training
        pretrain_config = test_config.copy()
        pretrain_config['training']['max_steps'] = 1
        pretrain_config['data']['num_samples'] = 3
        pretrain_config['output']['output_dir'] = os.path.join(temp_checkpoint_dir, "pretrain")
        
        finetune_config = test_config.copy()
        finetune_config['training']['max_epochs'] = 1
        finetune_config['data']['train_size'] = 3
        finetune_config['output']['output_dir'] = os.path.join(temp_checkpoint_dir, "finetune")
        
        # Create config files
        pretrain_config_path = os.path.join(temp_checkpoint_dir, "pretrain_config.json")
        finetune_config_path = os.path.join(temp_checkpoint_dir, "finetune_config.json")
        
        with open(pretrain_config_path, 'w') as f:
            json.dump(pretrain_config, f)
        with open(finetune_config_path, 'w') as f:
            json.dump(finetune_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            
            # Run pretraining
            run_pretraining(pretrain_config_path)
            
            # Verify pretraining checkpoint exists
            pretrain_dir = pretrain_config['output']['output_dir']
            pretrain_checkpoints = [f for f in os.listdir(pretrain_dir) if f.endswith('.pt')]
            assert len(pretrain_checkpoints) > 0
            
            # Run finetuning
            run_finetuning(finetune_config_path)
            
            # Verify finetuning checkpoint exists
            finetune_dir = finetune_config['output']['output_dir']
            finetune_checkpoints = [f for f in os.listdir(finetune_dir) if f.endswith('.pt')]
            assert len(finetune_checkpoints) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_finetune_to_evaluation_pipeline(self, test_config, temp_checkpoint_dir):
        """Test finetuning followed by evaluation"""
        # Configure for minimal training
        finetune_config = test_config.copy()
        finetune_config['training']['max_epochs'] = 1
        finetune_config['data']['train_size'] = 3
        finetune_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "finetune_eval_config.json")
        with open(config_path, 'w') as f:
            json.dump(finetune_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            
            # Run finetuning
            run_finetuning(config_path)
        
        # Find the best model checkpoint
        checkpoint_files = [f for f in os.listdir(temp_checkpoint_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) > 0
        
        # Try to find best model checkpoint
        best_checkpoint = None
        for checkpoint in checkpoint_files:
            if "best" in checkpoint.lower():
                best_checkpoint = checkpoint
                break
        
        if not best_checkpoint:
            best_checkpoint = checkpoint_files[0]  # Use any checkpoint
        
        checkpoint_path = os.path.join(temp_checkpoint_dir, best_checkpoint)
        
        # Run evaluation
        with patch('src.evaluation.DataProcessor') as mock_processor_class, \
             patch('src.evaluation.MultilingualTranslationModel') as mock_model_class:
            
            # Mock components to avoid model loading issues
            mock_model = MagicMock()
            mock_model.tokenizer = MagicMock()
            mock_model_class.return_value = mock_model
            
            mock_processor = MagicMock()
            mock_processor.load_wmt_en_ro.return_value = [
                {"source": "Hello", "target": "Bonjour"}
            ]
            mock_processor_class.return_value = mock_processor
            
            # Mock evaluator
            with patch('src.evaluation.TranslationEvaluator') as mock_evaluator_class:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_dataset.return_value = (
                    {'bleu': 15.0, 'rouge1': 0.4},
                    ["Salut"],
                    ["Bonjour"]
                )
                mock_evaluator.save_results = MagicMock()
                mock_evaluator_class.return_value = mock_evaluator
                
                results = evaluate_model(
                    model_path=checkpoint_path,
                    batch_size=2
                )
                
                # Verify evaluation ran successfully
                assert 'bleu' in results
                assert 'rouge1' in results

    @pytest.mark.integration
    def test_pipeline_checkpoint_continuity(self, test_config, temp_checkpoint_dir):
        """Test that checkpoints maintain continuity across pipeline stages"""
        # Configure for checkpoint testing
        test_config['training']['max_steps'] = 2
        test_config['training']['save_interval'] = 1
        test_config['data']['num_samples'] = 5
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "checkpoint_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            
            run_pretraining(config_path)
        
        # Load and verify checkpoint structure
        checkpoint_files = [f for f in os.listdir(temp_checkpoint_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) > 0
        
        checkpoint_path = os.path.join(temp_checkpoint_dir, checkpoint_files[0])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify checkpoint has required components for continuity
        required_components = [
            'model_state_dict',
            'optimizer_state_dict', 
            'step',
            'epoch',
            'best_val_loss'
        ]
        
        for component in required_components:
            assert component in checkpoint
            
        # Verify that state_dicts are not empty
        assert len(checkpoint['model_state_dict']) > 0
        assert len(checkpoint['optimizer_state_dict']) > 0

    @pytest.mark.integration
    @pytest.mark.parametrize("device_type", ["cpu"])  # Skip CUDA in CI
    def test_pipeline_device_consistency(self, test_config, temp_checkpoint_dir, device_type):
        """Test pipeline works consistently across devices"""
        device = torch.device(device_type)
        
        # Configure for minimal training
        test_config['training']['max_steps'] = 1
        test_config['data']['num_samples'] = 3
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        config_path = os.path.join(temp_checkpoint_dir, "device_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        with patch('src.trainer.wandb') as mock_wandb, \
             patch('src.trainer.torch.device') as mock_device:
            
            mock_wandb.init.return_value = MagicMock()
            mock_device.return_value = device
            
            # Should complete without device-related errors
            run_pretraining(config_path)
            
        # Verify checkpoint was created
        checkpoint_files = [f for f in os.listdir(temp_checkpoint_dir) if f.endswith('.pt')]
        assert len(checkpoint_files) > 0


class TestPipelineErrorHandling:
    """Test pipeline error handling and recovery"""

    @pytest.mark.integration
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write invalid JSON
            f.write("{ invalid json }")
            invalid_config_path = f.name
        
        try:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                run_pretraining(invalid_config_path)
        finally:
            os.unlink(invalid_config_path)

    @pytest.mark.integration
    def test_missing_config_file(self):
        """Test handling of missing configuration file"""
        non_existent_path = "/path/that/does/not/exist.json"
        
        with pytest.raises(FileNotFoundError):
            run_pretraining(non_existent_path)

    @pytest.mark.integration
    def test_incomplete_config_handling(self, temp_checkpoint_dir):
        """Test handling of incomplete configuration"""
        # Create config missing required fields
        incomplete_config = {
            "model": {"d_model": 128},
            # Missing training, data, and output sections
        }
        
        config_path = os.path.join(temp_checkpoint_dir, "incomplete_config.json")
        with open(config_path, 'w') as f:
            json.dump(incomplete_config, f)
        
        # Should handle missing sections gracefully with defaults
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            
            # May raise KeyError or provide defaults - either is acceptable
            try:
                run_pretraining(config_path)
            except KeyError:
                pass  # Expected for incomplete config

    @pytest.mark.integration
    def test_output_directory_creation(self, test_config):
        """Test that output directories are created automatically"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use non-existent subdirectory
            output_dir = os.path.join(temp_dir, "new_output_dir", "checkpoints")
            test_config['output']['output_dir'] = output_dir
            test_config['training']['max_steps'] = 1
            test_config['data']['num_samples'] = 3
            
            config_path = os.path.join(temp_dir, "auto_create_config.json")
            with open(config_path, 'w') as f:
                json.dump(test_config, f)
            
            with patch('src.trainer.wandb') as mock_wandb:
                mock_wandb.init.return_value = MagicMock()
                
                run_pretraining(config_path)
                
                # Directory should be created automatically
                assert os.path.exists(output_dir)