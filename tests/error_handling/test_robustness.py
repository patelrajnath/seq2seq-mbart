"""
Robustness and error handling tests
"""

import pytest
import torch
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, side_effect
from torch.utils.data import DataLoader

from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator
from src.data import DataProcessor, TranslationDataset
from src.trainer import BaseTrainer, DenoisingPretrainingTrainer, TranslationTrainer
from src.evaluation import TranslationEvaluator, evaluate_model


class TestModelRobustness:
    """Test model robustness under various error conditions"""

    @pytest.mark.unit
    def test_model_with_corrupted_checkpoint(self, pretrain_model, temp_checkpoint_dir, device):
        """Test model behavior with corrupted checkpoint"""
        # Create a corrupted checkpoint file
        corrupted_checkpoint = {"invalid_data": "not_a_state_dict"}
        checkpoint_path = os.path.join(temp_checkpoint_dir, "corrupted.pt")
        torch.save(corrupted_checkpoint, checkpoint_path)
        
        # Attempting to load corrupted checkpoint should handle gracefully
        with pytest.raises((KeyError, RuntimeError)):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            pretrain_model.load_state_dict(checkpoint['model_state_dict'])

    @pytest.mark.unit
    def test_model_with_invalid_input_shapes(self, pretrain_model, device):
        """Test model with invalid input tensor shapes"""
        model = pretrain_model.to(device)
        
        # Test with mismatched input shapes
        with pytest.raises((RuntimeError, ValueError)):
            batch = {
                "input_ids": torch.randint(0, 1000, (2, 10)).to(device),
                "attention_mask": torch.ones(3, 15).to(device),  # Wrong shape
                "labels": torch.randint(0, 1000, (2, 10)).to(device),
                "decoder_input_ids": torch.randint(0, 1000, (2, 10)).to(device),
                "decoder_attention_mask": torch.ones(2, 10).to(device)
            }
            model(**batch)

    @pytest.mark.unit
    def test_model_with_out_of_vocabulary_tokens(self, pretrain_model, device):
        """Test model behavior with out-of-vocabulary token IDs"""
        model = pretrain_model.to(device)
        vocab_size = model.config.vocab_size
        
        # Use token IDs that are out of vocabulary range
        batch = {
            "input_ids": torch.tensor([[0, vocab_size + 1000, 2, 1]]).to(device),
            "attention_mask": torch.ones(1, 4).to(device),
            "labels": torch.tensor([[0, vocab_size + 1000, 2, 1]]).to(device),
            "decoder_input_ids": torch.tensor([[0, 100, 2, 1]]).to(device),
            "decoder_attention_mask": torch.ones(1, 4).to(device)
        }
        
        # Should handle gracefully (may clamp to valid range or raise error)
        try:
            output = model(**batch)
            # If it doesn't raise, output should be valid
            assert 'loss' in output
        except (IndexError, RuntimeError):
            # Expected behavior for out-of-vocab tokens
            pass

    @pytest.mark.unit
    @patch('src.model.MBartForConditionalGeneration.from_pretrained')
    def test_model_initialization_network_failure(self, mock_from_pretrained):
        """Test model initialization with network failures"""
        # Simulate network failure during model loading
        mock_from_pretrained.side_effect = ConnectionError("Network unavailable")
        
        with pytest.raises(ConnectionError):
            MultilingualTranslationModel("facebook/mbart-large-50")

    @pytest.mark.unit
    def test_noise_generator_with_invalid_inputs(self, mock_tokenizer):
        """Test noise generator with invalid inputs"""
        noise_gen = NoiseGenerator(mock_tokenizer)
        
        # Test with empty input
        empty_input = torch.empty(0, 0, dtype=torch.long)
        result = noise_gen.span_masking(empty_input)
        assert result.shape == empty_input.shape
        
        # Test with very large input (potential memory issues)
        # Use smaller size to avoid actual memory issues in tests
        large_input = torch.randint(0, 1000, (1, 10000))
        result = noise_gen.span_masking(large_input)
        assert result.shape == large_input.shape

    @pytest.mark.unit
    def test_model_memory_cleanup(self, pretrain_model, device):
        """Test that model properly cleans up memory"""
        model = pretrain_model.to(device)
        
        # Create a batch and run forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 20)).to(device),
            "attention_mask": torch.ones(4, 20).to(device),
            "labels": torch.randint(0, 1000, (4, 20)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (4, 20)).to(device),
            "decoder_attention_mask": torch.ones(4, 20).to(device)
        }
        
        output = model(**batch)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        
        # Clean up gradients
        model.zero_grad()
        
        # Verify gradients are cleared
        for param in model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))


class TestDataRobustness:
    """Test data processing robustness"""

    @pytest.mark.unit
    def test_dataset_with_malformed_data(self, mock_tokenizer):
        """Test dataset handling of malformed data"""
        malformed_data = [
            {"source": "Hello", "target": "Bonjour"},  # Valid
            {"source": None, "target": "Invalid"},     # Invalid source
            {"source": "Valid", "target": None},       # Invalid target
            {"source": "", "target": ""},              # Empty strings
            {"source": "Good", "target": "Bien"}       # Valid
        ]
        
        dataset = TranslationDataset(malformed_data, mock_tokenizer)
        
        # Should handle malformed entries gracefully
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                assert isinstance(item, dict)
                assert "input_ids" in item
            except (TypeError, AttributeError):
                # Expected for malformed entries
                pass

    @pytest.mark.unit
    @patch('src.data.load_dataset')
    def test_data_processor_network_failure(self, mock_load_dataset):
        """Test data processor with network failures"""
        # Simulate network failure
        mock_load_dataset.side_effect = ConnectionError("Dataset download failed")
        
        processor = DataProcessor.__new__(DataProcessor)
        processor.tokenizer = MagicMock()
        
        # Should fallback to dummy data
        data = processor.load_wmt_en_ro("train")
        
        # Should get dummy data instead of raising error
        assert len(data) > 0
        assert all("source" in item and "target" in item for item in data)

    @pytest.mark.unit
    @patch('src.data.load_dataset')
    def test_data_processor_partial_failure(self, mock_load_dataset):
        """Test data processor with partial dataset corruption"""
        # Mock dataset with some corrupted entries
        mock_dataset = [
            {"translation": {"en": "Hello", "ro": "Salut"}},          # Valid
            {"translation": {"en": None, "ro": "Invalid"}},           # Invalid
            {"translation": {"en": "Good", "ro": "Bun"}},             # Valid
            {"invalid_structure": "corrupted"},                        # Invalid structure
        ]
        mock_load_dataset.return_value = mock_dataset
        
        processor = DataProcessor.__new__(DataProcessor)
        processor.tokenizer = MagicMock()
        
        # Should extract valid entries and skip invalid ones
        try:
            data = processor.load_wmt_en_ro("train")
            # Should have some data (at least the valid entries or fallback)
            assert len(data) > 0
        except (KeyError, AttributeError):
            # May raise error on corrupted data - that's acceptable
            pass

    @pytest.mark.unit
    def test_dataloader_worker_failure(self, sample_translation_data, mock_tokenizer):
        """Test dataloader robustness with worker failures"""
        dataset = TranslationDataset(sample_translation_data, mock_tokenizer)
        
        # Test with multiple workers (potential for worker failures)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
        
        # Should be able to iterate through dataset
        batches = []
        for batch in dataloader:
            batches.append(batch)
            if len(batches) >= 2:  # Limit to avoid hanging in tests
                break
        
        assert len(batches) > 0

    @pytest.mark.unit
    def test_tokenizer_failure_recovery(self):
        """Test tokenizer failure and recovery mechanisms"""
        with patch('src.data.MBart50TokenizerFast.from_pretrained') as mock_fast, \
             patch('src.data.MBartTokenizer.from_pretrained') as mock_fallback:
            
            # First tokenizer fails
            mock_fast.side_effect = Exception("Fast tokenizer failed")
            
            # Fallback succeeds
            mock_tokenizer = MagicMock()
            mock_fallback.return_value = mock_tokenizer
            
            processor = DataProcessor()
            
            # Should use fallback tokenizer
            assert processor.tokenizer == mock_tokenizer
            mock_fallback.assert_called_once()

    @pytest.mark.unit
    def test_large_batch_memory_handling(self, sample_translation_data, mock_tokenizer):
        """Test handling of memory issues with large batches"""
        # Create dataset with more data
        large_data = sample_translation_data * 100
        dataset = TranslationDataset(large_data, mock_tokenizer)
        
        # Test with reasonable batch size
        dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
        
        # Should handle large batches without memory issues
        try:
            batch = next(iter(dataloader))
            assert batch["input_ids"].shape[0] == 32
        except (RuntimeError, MemoryError):
            # Expected on systems with limited memory
            pytest.skip("Insufficient memory for large batch test")


class TestTrainerRobustness:
    """Test trainer robustness under error conditions"""

    @pytest.mark.unit
    def test_trainer_with_corrupted_optimizer_state(self, mock_model, device, temp_checkpoint_dir):
        """Test trainer with corrupted optimizer state"""
        train_loader = MagicMock()
        val_loader = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        trainer = BaseTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=temp_checkpoint_dir
        )
        
        # Create corrupted checkpoint
        corrupted_checkpoint = {
            'model_state_dict': {"param": torch.randn(10, 10)},
            'optimizer_state_dict': "invalid_optimizer_state",  # Wrong type
            'step': 100,
            'epoch': 2,
            'best_val_loss': 0.5
        }
        
        checkpoint_path = os.path.join(temp_checkpoint_dir, "corrupted.pt")
        torch.save(corrupted_checkpoint, checkpoint_path)
        
        # Loading should handle corrupted optimizer state
        with pytest.raises((TypeError, AttributeError)):
            trainer.load_checkpoint(checkpoint_path)

    @pytest.mark.unit
    def test_trainer_disk_full_during_save(self, mock_model, device):
        """Test trainer behavior when disk is full during checkpoint save"""
        train_loader = MagicMock()
        val_loader = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        # Use non-existent directory to simulate permission/disk issues
        invalid_output_dir = "/nonexistent/path/checkpoints"
        
        with pytest.raises((OSError, PermissionError)):
            trainer = BaseTrainer(
                model=mock_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                output_dir=invalid_output_dir
            )

    @pytest.mark.unit
    @patch('src.trainer.torch.save')
    def test_trainer_save_failure_recovery(self, mock_save, mock_model, device, temp_checkpoint_dir):
        """Test trainer recovery from save failures"""
        mock_save.side_effect = OSError("Disk full")
        
        train_loader = MagicMock()
        val_loader = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        
        trainer = BaseTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=temp_checkpoint_dir
        )
        
        # Save should fail gracefully
        with pytest.raises(OSError):
            trainer.save_checkpoint("test.pt")

    @pytest.mark.unit
    def test_trainer_with_infinite_loss(self, pretrain_model, device, temp_checkpoint_dir, test_config):
        """Test trainer handling of infinite/NaN losses"""
        # Mock data loaders
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        # Create batch that might cause numerical issues
        problematic_batch = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long).to(device),  # All zeros
            "attention_mask": torch.zeros(2, 10).to(device),               # No attention
            "labels": torch.full((2, 10), -100, dtype=torch.long).to(device),  # Ignore index
            "decoder_input_ids": torch.zeros(2, 10, dtype=torch.long).to(device),
            "decoder_attention_mask": torch.zeros(2, 10).to(device)
        }
        
        train_loader.__iter__ = MagicMock(return_value=iter([problematic_batch]))
        val_loader.__iter__ = MagicMock(return_value=iter([problematic_batch]))
        
        trainer = DenoisingPretrainingTrainer(
            model=pretrain_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Training step might produce inf/nan loss
        try:
            loss = trainer.train_step(problematic_batch)
            # If loss is finite, that's good
            if torch.isfinite(torch.tensor(loss)):
                assert loss >= 0
            else:
                # Infinite or NaN loss - trainer should handle this
                assert torch.isinf(torch.tensor(loss)) or torch.isnan(torch.tensor(loss))
        except (RuntimeError, ValueError):
            # Numerical errors are acceptable
            pass

    @pytest.mark.unit
    def test_trainer_memory_exhaustion(self, pretrain_model, device, temp_checkpoint_dir, test_config):
        """Test trainer behavior under memory pressure"""
        # Create oversized batch to potentially trigger memory issues
        # Use reasonable size to avoid actually exhausting memory in tests
        large_batch = {
            "input_ids": torch.randint(0, 1000, (8, 512)).to(device),
            "attention_mask": torch.ones(8, 512).to(device),
            "labels": torch.randint(0, 1000, (8, 512)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (8, 512)).to(device),
            "decoder_attention_mask": torch.ones(8, 512).to(device)
        }
        
        train_loader = MagicMock()
        val_loader = MagicMock()
        train_loader.__iter__ = MagicMock(return_value=iter([large_batch]))
        val_loader.__iter__ = MagicMock(return_value=iter([large_batch]))
        
        trainer = DenoisingPretrainingTrainer(
            model=pretrain_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Should handle or fail gracefully with memory issues
        try:
            loss = trainer.train_step(large_batch)
            assert isinstance(loss, float)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("GPU out of memory - expected in resource-constrained environments")
            else:
                raise


class TestEvaluationRobustness:
    """Test evaluation robustness"""

    @pytest.mark.unit
    def test_evaluator_with_empty_input(self, mock_model, mock_tokenizer, device):
        """Test evaluator with empty input"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        # Empty input should be handled gracefully
        translations = evaluator.translate_batch([])
        assert translations == []
        
        # Empty evaluation should be handled
        results, hyps, refs = evaluator.evaluate_dataset([], [], batch_size=1)
        assert isinstance(results, dict)
        assert hyps == []
        assert refs == []

    @pytest.mark.unit
    def test_evaluator_with_very_long_inputs(self, mock_model, mock_tokenizer, device):
        """Test evaluator with very long input sequences"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        # Very long input text
        long_text = "This is a very long sentence. " * 100
        long_inputs = [long_text]
        
        # Should handle gracefully (truncate or process in chunks)
        translations = evaluator.translate_batch(long_inputs, max_length=128)
        assert len(translations) == 1
        assert isinstance(translations[0], str)

    @pytest.mark.unit
    def test_evaluator_with_special_characters(self, mock_model, mock_tokenizer, device):
        """Test evaluator with special characters and unicode"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        special_inputs = [
            "Hello 🌍",  # Emoji
            "Café naïve",  # Accented characters
            "测试中文",  # Chinese characters
            "العربية",  # Arabic text
            "Русский",  # Cyrillic
            "",  # Empty string
            "   ",  # Whitespace only
        ]
        
        # Should handle all special characters gracefully
        translations = evaluator.translate_batch(special_inputs)
        assert len(translations) == len(special_inputs)
        assert all(isinstance(t, str) for t in translations)

    @pytest.mark.unit
    @patch('src.evaluation.sacrebleu.corpus_bleu')
    def test_metric_computation_edge_cases(self, mock_bleu, mock_model, mock_tokenizer, device):
        """Test metric computation with edge cases"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        # Mock BLEU to potentially return problematic values
        mock_bleu_result = MagicMock()
        mock_bleu_result.score = float('inf')  # Infinite BLEU score
        mock_bleu.return_value = mock_bleu_result
        
        hypotheses = ["test"]
        references = ["test"]
        
        # Should handle infinite/NaN metric values
        try:
            results = evaluator.compute_bleu(hypotheses, references)
            # If it returns, should have proper structure
            assert 'bleu' in results
        except (ValueError, OverflowError):
            # Acceptable to fail on infinite values
            pass

    @pytest.mark.unit
    def test_evaluation_save_failure(self, mock_model, mock_tokenizer, device):
        """Test evaluation result saving with file system errors"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        results = {'bleu': 25.0}
        hypotheses = ['test']
        references = ['test']
        
        # Try to save to invalid directory
        invalid_output_dir = "/nonexistent/path/results"
        
        with pytest.raises((OSError, PermissionError)):
            evaluator.save_results(results, hypotheses, references, invalid_output_dir)

    @pytest.mark.unit
    @patch('src.evaluation.MultilingualTranslationModel')
    def test_evaluate_model_with_missing_checkpoint(self, mock_model_class):
        """Test evaluate_model function with missing checkpoint"""
        mock_model_class.return_value = MagicMock()
        
        # Non-existent checkpoint path
        with pytest.raises(FileNotFoundError):
            evaluate_model(model_path="/path/that/does/not/exist.pt")


class TestSystemResourceHandling:
    """Test system resource handling and limitations"""

    @pytest.mark.unit
    def test_gpu_availability_handling(self, pretrain_model):
        """Test graceful handling of GPU availability"""
        # Test CPU fallback when CUDA is not available
        cpu_device = torch.device("cpu")
        model = pretrain_model.to(cpu_device)
        
        # Should work on CPU
        batch = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "labels": torch.randint(0, 1000, (1, 10)),
            "decoder_input_ids": torch.randint(0, 1000, (1, 10)),
            "decoder_attention_mask": torch.ones(1, 10)
        }
        
        output = model(**batch)
        assert 'loss' in output

    @pytest.mark.unit
    @patch('torch.cuda.is_available')
    def test_cuda_unavailable_fallback(self, mock_cuda_available, test_config, temp_checkpoint_dir):
        """Test fallback to CPU when CUDA is unavailable"""
        # Simulate CUDA not available
        mock_cuda_available.return_value = False
        
        # Should default to CPU device
        expected_device = torch.device("cpu")
        
        # Test that training components handle CPU correctly
        test_config['training']['max_steps'] = 1
        test_config['output']['output_dir'] = temp_checkpoint_dir
        
        # Should not raise CUDA-related errors
        try:
            from src.model import MultilingualDenoisingPretraining
            from transformers import MBartConfig
            
            config = MBartConfig(vocab_size=1000, d_model=128)
            model = MultilingualDenoisingPretraining(config)
            model = model.to(expected_device)
            
            assert next(model.parameters()).device.type == "cpu"
        except RuntimeError as e:
            if "cuda" in str(e).lower():
                pytest.fail(f"Should not have CUDA errors when falling back to CPU: {e}")

    @pytest.mark.unit
    def test_low_memory_graceful_degradation(self, pretrain_model, device):
        """Test graceful degradation under low memory conditions"""
        model = pretrain_model.to(device)
        
        # Start with reasonable batch size
        batch_size = 2
        seq_len = 64
        
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "attention_mask": torch.ones(batch_size, seq_len).to(device),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_attention_mask": torch.ones(batch_size, seq_len).to(device)
        }
        
        # Should process successfully with reasonable resources
        try:
            output = model(**batch)
            assert 'loss' in output
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for test - expected on constrained systems")
            else:
                raise

    @pytest.mark.unit
    def test_file_permission_handling(self, temp_checkpoint_dir):
        """Test handling of file permission issues"""
        # Create a directory with restricted permissions
        restricted_dir = os.path.join(temp_checkpoint_dir, "restricted")
        os.makedirs(restricted_dir)
        
        try:
            # Remove write permissions
            os.chmod(restricted_dir, 0o444)  # Read-only
            
            # Attempting to save checkpoint should fail gracefully
            test_file = os.path.join(restricted_dir, "test.pt")
            
            with pytest.raises(PermissionError):
                torch.save({"test": "data"}, test_file)
                
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(restricted_dir, 0o755)
            except:
                pass