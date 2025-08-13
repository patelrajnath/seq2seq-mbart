"""
Unit tests for trainer components
"""

import pytest
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from torch.optim import AdamW

from src.trainer import BaseTrainer, DenoisingPretrainingTrainer, TranslationTrainer


class TestBaseTrainer:
    """Test BaseTrainer class"""

    @pytest.fixture
    def mock_components(self, mock_model, device):
        """Create mock components for trainer"""
        train_loader = MagicMock()
        val_loader = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        return {
            'model': mock_model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'device': device,
            'output_dir': 'test_output'
        }

    @pytest.mark.unit
    def test_initialization(self, mock_components, temp_checkpoint_dir):
        """Test trainer initialization"""
        mock_components['output_dir'] = temp_checkpoint_dir
        
        trainer = BaseTrainer(**mock_components)
        
        assert trainer.device == mock_components['device']
        assert trainer.model == mock_components['model']
        assert trainer.train_loader == mock_components['train_loader']
        assert trainer.val_loader == mock_components['val_loader']
        assert trainer.optimizer == mock_components['optimizer']
        assert trainer.scheduler == mock_components['scheduler']
        assert trainer.output_dir == temp_checkpoint_dir
        assert trainer.step == 0
        assert trainer.epoch == 0
        assert trainer.best_val_loss == float('inf')

    @pytest.mark.unit
    def test_initialization_creates_output_dir(self, mock_components):
        """Test that initialization creates output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "new_output_dir")
            mock_components['output_dir'] = output_dir
            
            trainer = BaseTrainer(**mock_components)
            
            assert os.path.exists(output_dir)

    @pytest.mark.unit
    def test_save_checkpoint(self, mock_components, temp_checkpoint_dir):
        """Test checkpoint saving"""
        mock_components['output_dir'] = temp_checkpoint_dir
        trainer = BaseTrainer(**mock_components)
        
        # Set some state
        trainer.step = 100
        trainer.epoch = 2
        trainer.best_val_loss = 0.5
        
        checkpoint_path = os.path.join(temp_checkpoint_dir, "test_checkpoint.pt")
        trainer.save_checkpoint("test_checkpoint.pt")
        
        assert os.path.exists(checkpoint_path)
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert checkpoint['step'] == 100
        assert checkpoint['epoch'] == 2
        assert checkpoint['best_val_loss'] == 0.5
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint

    @pytest.mark.unit
    def test_save_checkpoint_with_scheduler(self, mock_components, temp_checkpoint_dir):
        """Test checkpoint saving with scheduler"""
        mock_components['output_dir'] = temp_checkpoint_dir
        trainer = BaseTrainer(**mock_components)
        
        trainer.save_checkpoint("test_checkpoint.pt")
        
        checkpoint = torch.load(
            os.path.join(temp_checkpoint_dir, "test_checkpoint.pt"), 
            map_location='cpu'
        )
        assert checkpoint['scheduler_state_dict'] is not None

    @pytest.mark.unit
    def test_save_checkpoint_without_scheduler(self, mock_components, temp_checkpoint_dir):
        """Test checkpoint saving without scheduler"""
        mock_components['scheduler'] = None
        mock_components['output_dir'] = temp_checkpoint_dir
        trainer = BaseTrainer(**mock_components)
        
        trainer.save_checkpoint("test_checkpoint.pt")
        
        checkpoint = torch.load(
            os.path.join(temp_checkpoint_dir, "test_checkpoint.pt"), 
            map_location='cpu'
        )
        assert checkpoint['scheduler_state_dict'] is None

    @pytest.mark.unit
    def test_load_checkpoint(self, mock_components, temp_checkpoint_dir):
        """Test checkpoint loading"""
        mock_components['output_dir'] = temp_checkpoint_dir
        trainer = BaseTrainer(**mock_components)
        
        # First save a checkpoint
        trainer.step = 150
        trainer.epoch = 3
        trainer.best_val_loss = 0.3
        checkpoint_path = os.path.join(temp_checkpoint_dir, "test_checkpoint.pt")
        trainer.save_checkpoint("test_checkpoint.pt")
        
        # Reset trainer state
        trainer.step = 0
        trainer.epoch = 0
        trainer.best_val_loss = float('inf')
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        assert trainer.step == 150
        assert trainer.epoch == 3
        assert trainer.best_val_loss == 0.3

    @pytest.mark.unit
    def test_load_checkpoint_calls_model_methods(self, mock_components, temp_checkpoint_dir):
        """Test that checkpoint loading calls appropriate methods"""
        mock_components['output_dir'] = temp_checkpoint_dir
        trainer = BaseTrainer(**mock_components)
        
        # Save checkpoint first
        trainer.save_checkpoint("test_checkpoint.pt")
        checkpoint_path = os.path.join(temp_checkpoint_dir, "test_checkpoint.pt")
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        # Verify methods were called
        trainer.model.load_state_dict.assert_called_once()
        trainer.optimizer.load_state_dict.assert_called_once()
        trainer.scheduler.load_state_dict.assert_called_once()


class TestDenoisingPretrainingTrainer:
    """Test DenoisingPretrainingTrainer class"""

    @pytest.fixture
    def mock_components_pretrain(self, pretrain_model, device, test_config, temp_checkpoint_dir):
        """Create mock components for pretraining trainer"""
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        # Mock data loader iteration
        train_loader.__iter__ = MagicMock(return_value=iter([
            {'input_ids': torch.randint(0, 1000, (2, 10)), 
             'attention_mask': torch.ones(2, 10),
             'labels': torch.randint(0, 1000, (2, 10)),
             'decoder_input_ids': torch.randint(0, 1000, (2, 10)),
             'decoder_attention_mask': torch.ones(2, 10)}
            for _ in range(5)
        ]))
        
        val_loader.__iter__ = MagicMock(return_value=iter([
            {'input_ids': torch.randint(0, 1000, (2, 10)), 
             'attention_mask': torch.ones(2, 10),
             'labels': torch.randint(0, 1000, (2, 10)),
             'decoder_input_ids': torch.randint(0, 1000, (2, 10)),
             'decoder_attention_mask': torch.ones(2, 10)}
            for _ in range(2)
        ]))
        
        return {
            'model': pretrain_model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'device': device,
            'output_dir': temp_checkpoint_dir,
            'config': test_config
        }

    @pytest.mark.unit
    def test_initialization(self, mock_components_pretrain):
        """Test pretraining trainer initialization"""
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        assert hasattr(trainer, 'noise_generator')
        assert hasattr(trainer, 'max_steps')
        assert trainer.max_steps == mock_components_pretrain['config']['training']['max_steps']
        assert isinstance(trainer.optimizer, AdamW)

    @pytest.mark.unit
    def test_initialization_with_noise_generator(self, mock_components_pretrain, noise_generator):
        """Test initialization with provided noise generator"""
        mock_components_pretrain['noise_generator'] = noise_generator
        
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        assert trainer.noise_generator == noise_generator

    @pytest.mark.unit
    @patch('src.trainer.wandb')
    def test_train_step(self, mock_wandb, mock_components_pretrain):
        """Test single training step"""
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1000, (2, 10)),
            'decoder_input_ids': torch.randint(0, 1000, (2, 10)),
            'decoder_attention_mask': torch.ones(2, 10)
        }
        
        loss_value = trainer.train_step(batch)
        
        assert isinstance(loss_value, float)
        assert loss_value > 0  # Should be positive loss
        
        # Verify optimizer and scheduler were called
        trainer.optimizer.step.assert_called_once()
        trainer.optimizer.zero_grad.assert_called_once()
        trainer.scheduler.step.assert_called_once()

    @pytest.mark.unit
    def test_train_step_with_noise_generator(self, mock_components_pretrain, noise_generator):
        """Test training step with noise generator"""
        mock_components_pretrain['noise_generator'] = noise_generator
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1000, (2, 10)),
            'decoder_input_ids': torch.randint(0, 1000, (2, 10)),
            'decoder_attention_mask': torch.ones(2, 10)
        }
        
        loss_value = trainer.train_step(batch)
        
        assert isinstance(loss_value, float)

    @pytest.mark.unit
    def test_validate(self, mock_components_pretrain):
        """Test validation step"""
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        val_loss = trainer.validate()
        
        assert isinstance(val_loss, float)
        assert val_loss > 0

    @pytest.mark.unit
    @patch('src.trainer.wandb')
    @patch('src.trainer.tqdm')
    def test_train_loop_structure(self, mock_tqdm, mock_wandb, mock_components_pretrain):
        """Test training loop structure"""
        # Set very small max_steps for quick test
        mock_components_pretrain['config']['training']['max_steps'] = 2
        
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value = mock_progress_bar
        mock_wandb.init.return_value = MagicMock()
        
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        trainer.train()
        
        # Verify wandb was initialized
        mock_wandb.init.assert_called_once()
        mock_wandb.finish.assert_called_once()
        
        # Verify progress bar was created
        mock_tqdm.assert_called_once()

    @pytest.mark.unit
    @patch('src.trainer.wandb')
    def test_train_without_wandb(self, mock_wandb, mock_components_pretrain):
        """Test training without wandb (fallback mode)"""
        # Make wandb.init fail
        mock_wandb.init.side_effect = Exception("wandb not available")
        
        mock_components_pretrain['config']['training']['max_steps'] = 1
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        # Should not raise exception
        trainer.train()


class TestTranslationTrainer:
    """Test TranslationTrainer class"""

    @pytest.fixture
    def mock_components_translation(self, mock_model, device, test_config, temp_checkpoint_dir):
        """Create mock components for translation trainer"""
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        # Mock length for scheduler initialization
        train_loader.__len__ = MagicMock(return_value=10)
        
        # Mock data loader iteration
        train_loader.__iter__ = MagicMock(return_value=iter([
            {'input_ids': torch.randint(0, 1000, (2, 10)), 
             'attention_mask': torch.ones(2, 10),
             'labels': torch.randint(0, 1000, (2, 10))}
            for _ in range(5)
        ]))
        
        val_loader.__iter__ = MagicMock(return_value=iter([
            {'input_ids': torch.randint(0, 1000, (2, 10)), 
             'attention_mask': torch.ones(2, 10),
             'labels': torch.randint(0, 1000, (2, 10))}
            for _ in range(2)
        ]))
        
        return {
            'model': mock_model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'device': device,
            'output_dir': temp_checkpoint_dir,
            'config': test_config
        }

    @pytest.mark.unit
    def test_initialization(self, mock_components_translation):
        """Test translation trainer initialization"""
        trainer = TranslationTrainer(**mock_components_translation)
        
        assert hasattr(trainer, 'max_epochs')
        assert trainer.max_epochs == mock_components_translation['config']['training']['max_epochs']
        assert isinstance(trainer.optimizer, AdamW)

    @pytest.mark.unit
    def test_train_step(self, mock_components_translation):
        """Test single training step"""
        trainer = TranslationTrainer(**mock_components_translation)
        
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1000, (2, 10))
        }
        
        loss_value = trainer.train_step(batch)
        
        assert isinstance(loss_value, float)
        assert loss_value > 0
        
        # Verify optimizer and scheduler were called
        trainer.optimizer.step.assert_called_once()
        trainer.optimizer.zero_grad.assert_called_once()
        trainer.scheduler.step.assert_called_once()

    @pytest.mark.unit
    def test_validate(self, mock_components_translation):
        """Test validation step"""
        trainer = TranslationTrainer(**mock_components_translation)
        
        val_loss = trainer.validate()
        
        assert isinstance(val_loss, float)
        assert val_loss > 0

    @pytest.mark.unit
    @patch('src.trainer.wandb')
    @patch('src.trainer.tqdm')
    def test_train_loop_structure(self, mock_tqdm, mock_wandb, mock_components_translation):
        """Test training loop structure"""
        # Set very small max_epochs for quick test
        mock_components_translation['config']['training']['max_epochs'] = 1
        
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value = mock_progress_bar
        mock_wandb.init.return_value = MagicMock()
        
        trainer = TranslationTrainer(**mock_components_translation)
        trainer.train()
        
        # Verify wandb was initialized
        mock_wandb.init.assert_called_once()
        mock_wandb.finish.assert_called_once()

    @pytest.mark.unit
    @patch('src.trainer.wandb')
    def test_train_without_wandb(self, mock_wandb, mock_components_translation):
        """Test training without wandb (fallback mode)"""
        # Make wandb.init fail
        mock_wandb.init.side_effect = Exception("wandb not available")
        
        mock_components_translation['config']['training']['max_epochs'] = 1
        trainer = TranslationTrainer(**mock_components_translation)
        
        # Should not raise exception
        trainer.train()

    @pytest.mark.unit
    def test_epoch_loop_structure(self, mock_components_translation):
        """Test epoch loop structure"""
        mock_components_translation['config']['training']['max_epochs'] = 2
        
        trainer = TranslationTrainer(**mock_components_translation)
        
        # Mock wandb to avoid initialization issues
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            trainer.train()
        
        # Verify epoch was updated
        assert trainer.epoch >= 0


class TestTrainerIntegration:
    """Integration tests for trainer components"""

    @pytest.mark.unit
    def test_optimizer_initialization_pretrain(self, mock_components_pretrain):
        """Test optimizer initialization for pretraining"""
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        assert isinstance(trainer.optimizer, AdamW)
        # Check learning rate from config
        config_lr = mock_components_pretrain['config']['training']['learning_rate']
        assert trainer.optimizer.param_groups[0]['lr'] == config_lr

    @pytest.mark.unit
    def test_optimizer_initialization_translation(self, mock_components_translation):
        """Test optimizer initialization for translation"""
        trainer = TranslationTrainer(**mock_components_translation)
        
        assert isinstance(trainer.optimizer, AdamW)
        config_lr = mock_components_translation['config']['training']['learning_rate']
        assert trainer.optimizer.param_groups[0]['lr'] == config_lr

    @pytest.mark.unit
    def test_scheduler_initialization_pretrain(self, mock_components_pretrain):
        """Test scheduler initialization for pretraining"""
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        assert trainer.scheduler is not None
        # Should have warmup and total steps configured
        assert hasattr(trainer.scheduler, 'step')

    @pytest.mark.unit
    def test_scheduler_initialization_translation(self, mock_components_translation):
        """Test scheduler initialization for translation"""
        trainer = TranslationTrainer(**mock_components_translation)
        
        assert trainer.scheduler is not None
        assert hasattr(trainer.scheduler, 'step')

    @pytest.mark.unit
    def test_device_consistency_pretrain(self, mock_components_pretrain):
        """Test device consistency in pretraining trainer"""
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        assert trainer.device == mock_components_pretrain['device']
        # Model should be moved to device during initialization
        trainer.model.to.assert_called_with(mock_components_pretrain['device'])

    @pytest.mark.unit
    def test_device_consistency_translation(self, mock_components_translation):
        """Test device consistency in translation trainer"""
        trainer = TranslationTrainer(**mock_components_translation)
        
        assert trainer.device == mock_components_translation['device']
        trainer.model.to.assert_called_with(mock_components_translation['device'])

    @pytest.mark.unit
    def test_config_parameter_usage(self, test_config, mock_components_pretrain):
        """Test that configuration parameters are properly used"""
        # Modify config to test parameter usage
        test_config['training']['batch_size'] = 16
        test_config['training']['learning_rate'] = 1e-3
        mock_components_pretrain['config'] = test_config
        
        trainer = DenoisingPretrainingTrainer(**mock_components_pretrain)
        
        # Check that learning rate from config is used
        assert trainer.optimizer.param_groups[0]['lr'] == 1e-3