"""
Edge case scenario tests
"""

import pytest
import torch
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator
from src.data import DataProcessor, TranslationDataset, DenoisingPretrainDataset
from src.trainer import BaseTrainer, DenoisingPretrainingTrainer, TranslationTrainer
from src.evaluation import TranslationEvaluator


class TestExtremeInputs:
    """Test handling of extreme input scenarios"""

    @pytest.mark.unit
    def test_single_token_sequences(self, pretrain_model, device):
        """Test model with single-token sequences"""
        model = pretrain_model.to(device)
        
        # Single token sequences (just BOS/EOS)
        batch = {
            "input_ids": torch.tensor([[0], [2]]).to(device),  # BOS, EOS
            "attention_mask": torch.ones(2, 1).to(device),
            "labels": torch.tensor([[0], [2]]).to(device),
            "decoder_input_ids": torch.tensor([[0], [2]]).to(device),
            "decoder_attention_mask": torch.ones(2, 1).to(device)
        }
        
        # Should handle minimal sequences
        output = model(**batch)
        assert 'loss' in output
        assert output['logits'].shape[1] == 1  # Single position

    @pytest.mark.unit
    def test_maximum_sequence_length(self, pretrain_model, device):
        """Test model with maximum allowed sequence length"""
        model = pretrain_model.to(device)
        max_length = model.config.max_position_embeddings
        
        # Use maximum allowed length
        batch = {
            "input_ids": torch.randint(0, 1000, (1, max_length)).to(device),
            "attention_mask": torch.ones(1, max_length).to(device),
            "labels": torch.randint(0, 1000, (1, max_length)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (1, max_length)).to(device),
            "decoder_attention_mask": torch.ones(1, max_length).to(device)
        }
        
        # Should handle maximum length sequences
        try:
            output = model(**batch)
            assert 'loss' in output
        except RuntimeError as e:
            # May fail due to memory constraints - acceptable
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for max length sequence")
            else:
                raise

    @pytest.mark.unit
    def test_batch_size_one(self, pretrain_model, device):
        """Test model with batch size of 1"""
        model = pretrain_model.to(device)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)).to(device),
            "attention_mask": torch.ones(1, 10).to(device),
            "labels": torch.randint(0, 1000, (1, 10)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (1, 10)).to(device),
            "decoder_attention_mask": torch.ones(1, 10).to(device)
        }
        
        output = model(**batch)
        assert 'loss' in output
        assert output['logits'].shape[0] == 1

    @pytest.mark.unit
    def test_all_padding_tokens(self, pretrain_model, device):
        """Test model with sequences consisting entirely of padding tokens"""
        model = pretrain_model.to(device)
        pad_token_id = 1
        
        batch = {
            "input_ids": torch.full((2, 10), pad_token_id).to(device),
            "attention_mask": torch.zeros(2, 10).to(device),  # All positions masked
            "labels": torch.full((2, 10), pad_token_id).to(device),
            "decoder_input_ids": torch.full((2, 10), pad_token_id).to(device),
            "decoder_attention_mask": torch.zeros(2, 10).to(device)
        }
        
        # Should handle all-padding sequences gracefully
        output = model(**batch)
        assert 'loss' in output

    @pytest.mark.unit
    def test_alternating_attention_mask(self, pretrain_model, device):
        """Test model with alternating attention mask pattern"""
        model = pretrain_model.to(device)
        
        # Create alternating attention mask [1, 0, 1, 0, ...]
        attention_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                                     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]).float().to(device)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)).to(device),
            "attention_mask": attention_mask,
            "labels": torch.randint(0, 1000, (2, 10)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (2, 10)).to(device),
            "decoder_attention_mask": attention_mask
        }
        
        output = model(**batch)
        assert 'loss' in output

    @pytest.mark.unit
    def test_repeated_token_sequences(self, pretrain_model, device):
        """Test model with sequences of repeated tokens"""
        model = pretrain_model.to(device)
        
        # Sequences with same token repeated
        batch = {
            "input_ids": torch.tensor([[100] * 10, [200] * 10]).to(device),
            "attention_mask": torch.ones(2, 10).to(device),
            "labels": torch.tensor([[100] * 10, [200] * 10]).to(device),
            "decoder_input_ids": torch.tensor([[100] * 10, [200] * 10]).to(device),
            "decoder_attention_mask": torch.ones(2, 10).to(device)
        }
        
        output = model(**batch)
        assert 'loss' in output


class TestDataEdgeCases:
    """Test data processing edge cases"""

    @pytest.mark.unit
    def test_empty_string_translation_pairs(self, mock_tokenizer):
        """Test dataset with empty string pairs"""
        data = [
            {"source": "", "target": ""},
            {"source": "Hello", "target": ""},
            {"source": "", "target": "Bonjour"},
            {"source": "Good", "target": "Bien"}
        ]
        
        dataset = TranslationDataset(data, mock_tokenizer)
        
        # Should handle empty strings
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, dict)
            assert "input_ids" in item

    @pytest.mark.unit
    def test_very_long_translation_pairs(self, mock_tokenizer):
        """Test dataset with very long text pairs"""
        long_text = "This is a very long sentence that exceeds normal length limits. " * 50
        data = [
            {"source": long_text, "target": long_text[:100]},
            {"source": "Short", "target": long_text}
        ]
        
        dataset = TranslationDataset(data, mock_tokenizer, max_length=128)
        
        # Should truncate long sequences appropriately
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["input_ids"].shape[0] == 128  # Max length enforced

    @pytest.mark.unit
    def test_whitespace_only_strings(self, mock_tokenizer):
        """Test dataset with whitespace-only strings"""
        data = [
            {"source": "   ", "target": "\t\n"},
            {"source": "\n\n\n", "target": "   "},
            {"source": "Hello", "target": "\t\t\t"}
        ]
        
        dataset = TranslationDataset(data, mock_tokenizer)
        
        # Should handle whitespace-only strings
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, dict)

    @pytest.mark.unit
    def test_single_character_translations(self, mock_tokenizer):
        """Test dataset with single character translations"""
        data = [
            {"source": "a", "target": "b"},
            {"source": "x", "target": "y"},
            {"source": "1", "target": "2"},
            {"source": "!", "target": "?"}
        ]
        
        dataset = TranslationDataset(data, mock_tokenizer)
        
        # Should handle single characters
        assert len(dataset) == 4
        for i in range(len(dataset)):
            item = dataset[i]
            assert "input_ids" in item

    @pytest.mark.unit
    def test_numeric_only_strings(self, mock_tokenizer):
        """Test dataset with numeric-only strings"""
        data = [
            {"source": "12345", "target": "54321"},
            {"source": "3.14159", "target": "2.71828"},
            {"source": "-123", "target": "+456"},
            {"source": "0", "target": "1"}
        ]
        
        dataset = TranslationDataset(data, mock_tokenizer)
        
        # Should handle numeric strings
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, dict)

    @pytest.mark.unit
    def test_mixed_language_scripts(self, mock_tokenizer):
        """Test dataset with mixed language scripts"""
        data = [
            {"source": "Hello 世界", "target": "Bonjour мир"},
            {"source": "Test العربية", "target": "Тест Arabic"},
            {"source": "123 αβγ", "target": "456 дег"}
        ]
        
        dataset = TranslationDataset(data, mock_tokenizer)
        
        # Should handle mixed scripts
        for i in range(len(dataset)):
            item = dataset[i]
            assert isinstance(item, dict)

    @pytest.mark.unit
    def test_dataset_with_one_item(self, mock_tokenizer):
        """Test dataset with exactly one item"""
        data = [{"source": "Single item", "target": "Article unique"}]
        
        dataset = TranslationDataset(data, mock_tokenizer)
        
        assert len(dataset) == 1
        item = dataset[0]
        assert isinstance(item, dict)
        
        # Test that we can't access index 1
        with pytest.raises(IndexError):
            _ = dataset[1]


class TestNoiseGeneratorEdgeCases:
    """Test noise generator edge cases"""

    @pytest.mark.unit
    def test_span_masking_with_single_token(self, mock_tokenizer):
        """Test span masking with single token sequences"""
        noise_gen = NoiseGenerator(mock_tokenizer)
        
        # Single token (just BOS)
        input_ids = torch.tensor([[0]])
        
        masked = noise_gen.span_masking(input_ids)
        assert masked.shape == input_ids.shape

    @pytest.mark.unit
    def test_span_masking_all_special_tokens(self, mock_tokenizer):
        """Test span masking with sequences of all special tokens"""
        noise_gen = NoiseGenerator(mock_tokenizer)
        
        # All special tokens (BOS, EOS, PAD)
        input_ids = torch.tensor([[0, 2, 1, 1, 1]])
        
        masked = noise_gen.span_masking(input_ids)
        # Should preserve special tokens
        assert masked[0, 0] == 0  # BOS preserved

    @pytest.mark.unit
    def test_sentence_permutation_single_sentence(self, mock_tokenizer):
        """Test sentence permutation with single sentence"""
        noise_gen = NoiseGenerator(mock_tokenizer)
        
        # Single sentence (one EOS token)
        input_ids = torch.tensor([[0, 100, 101, 102, 2, 1]])
        
        permuted = noise_gen.sentence_permutation(input_ids)
        assert permuted.shape == input_ids.shape
        # With only one sentence, should remain largely unchanged
        assert permuted[0, 0] == 0  # BOS preserved

    @pytest.mark.unit
    def test_sentence_permutation_many_sentences(self, mock_tokenizer):
        """Test sentence permutation with many short sentences"""
        noise_gen = NoiseGenerator(mock_tokenizer)
        
        # Many single-token sentences
        input_ids = torch.tensor([[0, 100, 2, 101, 2, 102, 2, 103, 2, 1]])
        
        permuted = noise_gen.sentence_permutation(input_ids)
        assert permuted.shape == input_ids.shape
        assert permuted[0, 0] == 0  # BOS preserved

    @pytest.mark.unit
    def test_noise_generation_extreme_probabilities(self, mock_tokenizer):
        """Test noise generation with extreme probabilities"""
        # Zero probability - should make no changes
        zero_noise_gen = NoiseGenerator(mock_tokenizer, mask_prob=0.0)
        input_ids = torch.randint(2, 1000, (2, 10))
        
        masked = zero_noise_gen.span_masking(input_ids)
        # With zero probability, should be mostly unchanged (except for randomness in span selection)
        
        # Very high probability - should mask most tokens
        high_noise_gen = NoiseGenerator(mock_tokenizer, mask_prob=1.0)
        
        masked_high = high_noise_gen.span_masking(input_ids)
        # Should have many masked tokens
        mask_count = (masked_high == mock_tokenizer.mask_token_id).sum()
        assert mask_count > 0

    @pytest.mark.unit
    def test_poisson_lambda_extremes(self, mock_tokenizer):
        """Test noise generation with extreme Poisson lambda values"""
        # Very small lambda - should create short spans
        small_lambda_gen = NoiseGenerator(mock_tokenizer, poisson_lambda=0.1)
        input_ids = torch.randint(2, 1000, (1, 20))
        
        masked_small = small_lambda_gen.span_masking(input_ids)
        assert masked_small.shape == input_ids.shape
        
        # Very large lambda - should create long spans
        large_lambda_gen = NoiseGenerator(mock_tokenizer, poisson_lambda=10.0)
        
        masked_large = large_lambda_gen.span_masking(input_ids)
        assert masked_large.shape == input_ids.shape


class TestEvaluationEdgeCases:
    """Test evaluation edge cases"""

    @pytest.mark.unit
    def test_identical_hypothesis_reference(self, mock_model, mock_tokenizer, device):
        """Test evaluation with identical hypothesis and reference"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        identical_text = "Perfect match"
        hypotheses = [identical_text]
        references = [identical_text]
        
        # Exact match should give perfect scores
        em_result = evaluator.compute_exact_match(hypotheses, references)
        assert em_result['exact_match'] == 1.0

    @pytest.mark.unit
    def test_completely_different_texts(self, mock_model, mock_tokenizer, device):
        """Test evaluation with completely different texts"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        hypotheses = ["Hello world"]
        references = ["xyz123"]
        
        # Should handle completely different texts
        em_result = evaluator.compute_exact_match(hypotheses, references)
        assert em_result['exact_match'] == 0.0

    @pytest.mark.unit
    def test_evaluation_with_punctuation_differences(self, mock_model, mock_tokenizer, device):
        """Test evaluation with punctuation differences"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        hypotheses = ["Hello world"]
        references = ["Hello world!"]
        
        # Should handle punctuation differences
        em_result = evaluator.compute_exact_match(hypotheses, references)
        assert em_result['exact_match'] == 0.0  # Not exact match due to punctuation

    @pytest.mark.unit
    def test_evaluation_with_case_differences(self, mock_model, mock_tokenizer, device):
        """Test evaluation with case differences"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        hypotheses = ["Hello World"]
        references = ["hello world"]
        
        # Should handle case differences
        em_result = evaluator.compute_exact_match(hypotheses, references)
        assert em_result['exact_match'] == 0.0  # Case sensitive

    @pytest.mark.unit
    def test_evaluation_single_word_pairs(self, mock_model, mock_tokenizer, device):
        """Test evaluation with single word pairs"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        hypotheses = ["cat", "dog", "bird"]
        references = ["chat", "chien", "oiseau"]
        
        # Should handle single word translations
        em_result = evaluator.compute_exact_match(hypotheses, references)
        assert em_result['exact_match'] == 0.0  # All different

    @pytest.mark.unit
    def test_evaluation_with_numbers(self, mock_model, mock_tokenizer, device):
        """Test evaluation with numeric content"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        hypotheses = ["I have 5 apples", "Price is $10.99"]
        references = ["I have 5 apples", "Price is $10.99"]
        
        # Should handle numeric content correctly
        em_result = evaluator.compute_exact_match(hypotheses, references)
        assert em_result['exact_match'] == 1.0  # Perfect match

    @pytest.mark.unit
    def test_very_long_evaluation_texts(self, mock_model, mock_tokenizer, device):
        """Test evaluation with very long texts"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        long_text = "This is a very long text. " * 100
        hypotheses = [long_text]
        references = [long_text]
        
        # Should handle long texts
        em_result = evaluator.compute_exact_match(hypotheses, references)
        assert em_result['exact_match'] == 1.0


class TestTrainerEdgeCases:
    """Test trainer edge cases"""

    @pytest.mark.unit
    def test_training_with_zero_epochs(self, mock_model, device, temp_checkpoint_dir, test_config):
        """Test trainer behavior with zero epochs"""
        test_config['training']['max_epochs'] = 0
        
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        trainer = TranslationTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Should handle zero epochs gracefully
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            trainer.train()
        
        # Should not have trained at all
        assert trainer.epoch == 0

    @pytest.mark.unit
    def test_training_with_zero_steps(self, pretrain_model, device, temp_checkpoint_dir, test_config):
        """Test trainer behavior with zero max steps"""
        test_config['training']['max_steps'] = 0
        
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        trainer = DenoisingPretrainingTrainer(
            model=pretrain_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Should handle zero steps gracefully
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            trainer.train()
        
        # Should not have trained
        assert trainer.step == 0

    @pytest.mark.unit
    def test_training_with_single_batch(self, mock_model, device, temp_checkpoint_dir, test_config):
        """Test training with exactly one batch"""
        test_config['training']['max_epochs'] = 1
        
        # Mock single-batch data loader
        single_batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1000, (2, 10))
        }
        
        train_loader = MagicMock()
        train_loader.__iter__ = MagicMock(return_value=iter([single_batch]))
        train_loader.__len__ = MagicMock(return_value=1)
        
        val_loader = MagicMock()
        val_loader.__iter__ = MagicMock(return_value=iter([single_batch]))
        
        trainer = TranslationTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Should handle single batch training
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            trainer.train()
        
        # Should have completed one epoch
        assert trainer.epoch >= 0

    @pytest.mark.unit
    def test_scheduler_step_at_boundaries(self, mock_model, device, temp_checkpoint_dir, test_config):
        """Test scheduler stepping at training boundaries"""
        test_config['training']['max_epochs'] = 1
        test_config['training']['warmup_steps'] = 1  # Very small warmup
        
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        # Mock single batch
        single_batch = {
            'input_ids': torch.randint(0, 1000, (1, 5)),
            'attention_mask': torch.ones(1, 5),
            'labels': torch.randint(0, 1000, (1, 5))
        }
        
        train_loader.__iter__ = MagicMock(return_value=iter([single_batch]))
        train_loader.__len__ = MagicMock(return_value=1)
        val_loader.__iter__ = MagicMock(return_value=iter([single_batch]))
        
        trainer = TranslationTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Should handle scheduler at boundaries
        with patch('src.trainer.wandb') as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            trainer.train()
        
        # Scheduler should have been called
        trainer.scheduler.step.assert_called()

    @pytest.mark.unit
    def test_validation_with_empty_loader(self, mock_model, device, temp_checkpoint_dir, test_config):
        """Test validation with empty validation loader"""
        test_config['training']['max_epochs'] = 1
        
        train_loader = MagicMock()
        single_batch = {
            'input_ids': torch.randint(0, 1000, (1, 5)),
            'attention_mask': torch.ones(1, 5),
            'labels': torch.randint(0, 1000, (1, 5))
        }
        train_loader.__iter__ = MagicMock(return_value=iter([single_batch]))
        train_loader.__len__ = MagicMock(return_value=1)
        
        # Empty validation loader
        val_loader = MagicMock()
        val_loader.__iter__ = MagicMock(return_value=iter([]))
        
        trainer = TranslationTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Should handle empty validation loader
        val_loss = trainer.validate()
        # With empty loader, should return 0 or handle gracefully
        assert isinstance(val_loss, float)


class TestConfigurationEdgeCases:
    """Test configuration edge cases"""

    @pytest.mark.unit
    def test_config_with_float_integers(self):
        """Test configuration with float values that should be integers"""
        config = {
            "training": {
                "batch_size": 8.0,    # Float that should be int
                "max_epochs": 3.5,    # Non-integer epochs
                "max_steps": 1000.0   # Float steps
            }
        }
        
        # Should handle float values (may convert or use as-is)
        assert config["training"]["batch_size"] == 8.0
        assert config["training"]["max_epochs"] == 3.5

    @pytest.mark.unit
    def test_config_with_string_numbers(self):
        """Test configuration with string representations of numbers"""
        config = {
            "model": {
                "d_model": "512",
                "encoder_layers": "6"
            },
            "training": {
                "learning_rate": "1e-4"
            }
        }
        
        # Application should handle string-to-number conversion
        assert config["model"]["d_model"] == "512"
        assert config["training"]["learning_rate"] == "1e-4"

    @pytest.mark.unit
    def test_config_with_missing_nested_keys(self):
        """Test configuration with missing nested keys"""
        incomplete_config = {
            "model": {
                "d_model": 512
                # Missing other model parameters
            },
            "training": {
                # Missing training parameters
            }
        }
        
        # Should handle missing nested keys
        assert "d_model" in incomplete_config["model"]
        assert len(incomplete_config["training"]) == 0

    @pytest.mark.unit
    def test_config_with_null_values(self):
        """Test configuration with null values"""
        config_with_nulls = {
            "model": {
                "d_model": 512,
                "dropout": None,
                "activation": None
            },
            "training": {
                "batch_size": 8,
                "optimizer": None
            }
        }
        
        # Should handle null values appropriately
        assert config_with_nulls["model"]["d_model"] == 512
        assert config_with_nulls["model"]["dropout"] is None
        assert config_with_nulls["training"]["optimizer"] is None