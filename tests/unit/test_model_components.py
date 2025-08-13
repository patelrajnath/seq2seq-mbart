"""
Unit tests for model components
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from transformers import MBartConfig

from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator


class TestMultilingualDenoisingPretraining:
    """Test MultilingualDenoisingPretraining class"""

    @pytest.mark.unit
    def test_initialization(self, small_mbart_config):
        """Test model initialization"""
        model = MultilingualDenoisingPretraining(small_mbart_config)
        
        assert model.config == small_mbart_config
        assert hasattr(model, 'model')
        assert model.tokenizer is None
        assert isinstance(model, nn.Module)

    @pytest.mark.unit
    def test_forward_pass(self, pretrain_model, sample_batch, device):
        """Test forward pass"""
        model = pretrain_model.to(device)
        batch = {k: v.to(device) for k, v in sample_batch.items()}
        
        outputs = model(**batch)
        
        assert 'loss' in outputs
        assert 'logits' in outputs
        assert 'last_hidden_state' in outputs
        
        # Check output shapes
        assert outputs['loss'].dim() == 0  # scalar loss
        assert outputs['logits'].shape[0] == batch['input_ids'].shape[0]  # batch size
        assert outputs['logits'].shape[2] == model.config.vocab_size  # vocab size

    @pytest.mark.unit
    def test_parameter_count(self, pretrain_model):
        """Test parameter counting"""
        total_params = sum(p.numel() for p in pretrain_model.parameters())
        trainable_params = sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable

    @pytest.mark.unit
    def test_device_handling(self, pretrain_model, all_devices):
        """Test model device handling"""
        model = pretrain_model.to(all_devices)
        
        # Check that model parameters are on correct device
        for param in model.parameters():
            assert param.device.type == all_devices.type

    @pytest.mark.unit
    def test_gradient_flow(self, pretrain_model, sample_batch, device):
        """Test gradient flow through model"""
        model = pretrain_model.to(device)
        batch = {k: v.to(device) for k, v in sample_batch.items()}
        
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMultilingualTranslationModel:
    """Test MultilingualTranslationModel class"""

    @pytest.mark.unit
    @patch('src.model.MBartForConditionalGeneration.from_pretrained')
    @patch('src.model.MBart50TokenizerFast.from_pretrained')
    def test_initialization_success(self, mock_tokenizer, mock_model):
        """Test successful model initialization"""
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        model = MultilingualTranslationModel("facebook/mbart-large-50")
        
        assert hasattr(model, 'model')
        assert hasattr(model, 'tokenizer')
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    @pytest.mark.unit
    @patch('src.model.MBartForConditionalGeneration.from_pretrained')
    @patch('src.model.MBart50TokenizerFast.from_pretrained')
    @patch('src.model.MBartTokenizer.from_pretrained')
    def test_tokenizer_fallback(self, mock_fallback_tokenizer, mock_tokenizer, mock_model):
        """Test tokenizer fallback mechanism"""
        mock_model.return_value = MagicMock()
        mock_tokenizer.side_effect = Exception("Fast tokenizer failed")
        mock_fallback_tokenizer.return_value = MagicMock()
        
        model = MultilingualTranslationModel("facebook/mbart-large-50")
        
        mock_fallback_tokenizer.assert_called_once()

    @pytest.mark.unit
    def test_forward_pass(self, mock_model, sample_batch, device):
        """Test forward pass"""
        # Create model with mock
        model = MultilingualTranslationModel.__new__(MultilingualTranslationModel)
        model.model = mock_model
        model.tokenizer = MagicMock()
        
        model = model.to(device)
        batch = {k: v.to(device) for k, v in sample_batch.items()}
        
        outputs = model(**batch)
        
        assert 'loss' in outputs
        assert 'logits' in outputs

    @pytest.mark.unit
    def test_generate_method(self, mock_model, device):
        """Test generate method"""
        # Create model with mock
        model = MultilingualTranslationModel.__new__(MultilingualTranslationModel)
        model.model = mock_model
        model.tokenizer = MagicMock()
        
        input_ids = torch.randint(0, 1000, (2, 10)).to(device)
        attention_mask = torch.ones(2, 10).to(device)
        
        # Mock the generate method
        mock_model.generate.return_value = torch.randint(0, 1000, (2, 15))
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=20,
            num_beams=3
        )
        
        mock_model.generate.assert_called_once()
        assert outputs.shape[0] == 2  # batch size
        assert outputs.shape[1] == 15  # generated length

    @pytest.mark.unit
    def test_generate_with_default_params(self, mock_model, device):
        """Test generate with default parameters"""
        model = MultilingualTranslationModel.__new__(MultilingualTranslationModel)
        model.model = mock_model
        model.tokenizer = MagicMock()
        
        input_ids = torch.randint(0, 1000, (1, 10)).to(device)
        attention_mask = torch.ones(1, 10).to(device)
        
        mock_model.generate.return_value = torch.randint(0, 1000, (1, 20))
        
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check that default parameters were used
        call_args = mock_model.generate.call_args
        assert call_args.kwargs['max_length'] == 128
        assert call_args.kwargs['num_beams'] == 5
        assert call_args.kwargs['length_penalty'] == 1.0
        assert call_args.kwargs['early_stopping'] == True


class TestNoiseGenerator:
    """Test NoiseGenerator class"""

    @pytest.mark.unit
    def test_initialization(self, mock_tokenizer):
        """Test noise generator initialization"""
        noise_gen = NoiseGenerator(mock_tokenizer)
        
        assert noise_gen.tokenizer == mock_tokenizer
        assert noise_gen.mask_prob == 0.35
        assert noise_gen.poisson_lambda == 3.5
        assert noise_gen.mask_token_id == mock_tokenizer.mask_token_id

    @pytest.mark.unit
    def test_initialization_custom_params(self, mock_tokenizer):
        """Test noise generator with custom parameters"""
        noise_gen = NoiseGenerator(
            mock_tokenizer, 
            mask_prob=0.5, 
            poisson_lambda=2.0
        )
        
        assert noise_gen.mask_prob == 0.5
        assert noise_gen.poisson_lambda == 2.0

    @pytest.mark.unit
    def test_span_masking(self, noise_generator):
        """Test span masking functionality"""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(2, 1000, (batch_size, seq_len))  # Avoid special tokens
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        masked_input = noise_generator.span_masking(input_ids)
        
        assert masked_input.shape == input_ids.shape
        assert not torch.equal(masked_input, input_ids)  # Should be different
        
        # Check that some tokens were masked
        mask_token_id = noise_generator.mask_token_id
        assert (masked_input == mask_token_id).any()

    @pytest.mark.unit
    def test_span_masking_preserves_special_tokens(self, noise_generator):
        """Test that span masking preserves special tokens"""
        input_ids = torch.tensor([[0, 100, 200, 300, 2, 1, 1]])  # <s>, tokens, </s>, <pad>
        
        masked_input = noise_generator.span_masking(input_ids)
        
        # First token (BOS) should be preserved
        assert masked_input[0, 0] == 0
        # Padding tokens should be preserved
        assert (masked_input[0, -2:] == 1).all()

    @pytest.mark.unit
    def test_sentence_permutation(self, noise_generator):
        """Test sentence permutation functionality"""
        # Create input with sentence boundaries (EOS tokens)
        input_ids = torch.tensor([[0, 100, 101, 2, 200, 201, 2, 1, 1]])  # Two sentences
        
        torch.manual_seed(42)
        permuted_input = noise_generator.sentence_permutation(input_ids)
        
        assert permuted_input.shape == input_ids.shape
        # First token (BOS) should be preserved
        assert permuted_input[0, 0] == 0

    @pytest.mark.unit
    def test_sentence_permutation_no_sentences(self, noise_generator):
        """Test sentence permutation with no sentence boundaries"""
        input_ids = torch.tensor([[0, 100, 101, 102, 1, 1]])  # No EOS tokens
        
        permuted_input = noise_generator.sentence_permutation(input_ids)
        
        # Should return unchanged input when no sentence boundaries
        assert torch.equal(permuted_input, input_ids)

    @pytest.mark.unit
    def test_add_noise_span_masking(self, noise_generator):
        """Test add_noise with span masking"""
        input_ids = torch.randint(2, 1000, (2, 10))
        
        noisy_input = noise_generator.add_noise(input_ids, noise_type="span_masking")
        
        assert noisy_input.shape == input_ids.shape
        assert not torch.equal(noisy_input, input_ids)

    @pytest.mark.unit
    def test_add_noise_sentence_permutation(self, noise_generator):
        """Test add_noise with sentence permutation"""
        input_ids = torch.tensor([[0, 100, 2, 200, 2, 1, 1]])
        
        noisy_input = noise_generator.add_noise(input_ids, noise_type="sentence_permutation")
        
        assert noisy_input.shape == input_ids.shape

    @pytest.mark.unit
    def test_add_noise_both(self, noise_generator):
        """Test add_noise with both noise types"""
        input_ids = torch.tensor([[0, 100, 2, 200, 2, 1, 1]])
        
        noisy_input = noise_generator.add_noise(input_ids, noise_type="both")
        
        assert noisy_input.shape == input_ids.shape

    @pytest.mark.unit
    def test_add_noise_invalid_type(self, noise_generator):
        """Test add_noise with invalid noise type"""
        input_ids = torch.randint(2, 1000, (2, 10))
        
        noisy_input = noise_generator.add_noise(input_ids, noise_type="invalid")
        
        # Should return unchanged input for invalid noise type
        assert torch.equal(noisy_input, input_ids)

    @pytest.mark.unit
    def test_empty_input_handling(self, noise_generator):
        """Test handling of empty inputs"""
        empty_input = torch.empty(0, 0, dtype=torch.long)
        
        # Should handle empty input gracefully
        masked_input = noise_generator.span_masking(empty_input)
        assert masked_input.shape == empty_input.shape

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [5, 10, 20])
    def test_noise_with_different_sizes(self, noise_generator, batch_size, seq_len):
        """Test noise generation with different input sizes"""
        input_ids = torch.randint(2, 1000, (batch_size, seq_len))
        
        masked_input = noise_generator.span_masking(input_ids)
        
        assert masked_input.shape == (batch_size, seq_len)
        # Should have some differences for non-trivial inputs
        if seq_len > 2:
            assert not torch.equal(masked_input, input_ids)

    @pytest.mark.unit
    def test_mask_probability_effect(self, mock_tokenizer):
        """Test effect of different masking probabilities"""
        high_prob_gen = NoiseGenerator(mock_tokenizer, mask_prob=0.8)
        low_prob_gen = NoiseGenerator(mock_tokenizer, mask_prob=0.1)
        
        input_ids = torch.randint(2, 1000, (1, 20))
        
        torch.manual_seed(42)
        high_masked = high_prob_gen.span_masking(input_ids)
        
        torch.manual_seed(42)
        low_masked = low_prob_gen.span_masking(input_ids)
        
        # Higher probability should result in more masking
        high_mask_count = (high_masked == mock_tokenizer.mask_token_id).sum()
        low_mask_count = (low_masked == mock_tokenizer.mask_token_id).sum()
        
        assert high_mask_count >= low_mask_count

    @pytest.mark.unit
    def test_deterministic_behavior_with_seed(self, noise_generator):
        """Test deterministic behavior with fixed seed"""
        input_ids = torch.randint(2, 1000, (2, 10))
        
        torch.manual_seed(42)
        masked1 = noise_generator.span_masking(input_ids)
        
        torch.manual_seed(42)
        masked2 = noise_generator.span_masking(input_ids)
        
        assert torch.equal(masked1, masked2)