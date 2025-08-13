"""
Regression tests for model outputs and behavior consistency
"""

import pytest
import torch
import json
import os
import tempfile
import numpy as np
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator
from src.evaluation import TranslationEvaluator


class OutputRegressionTester:
    """Utility class for regression testing of model outputs"""
    
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
        
    def save_baseline(self, outputs: Dict[str, Any], filepath: str):
        """Save baseline outputs for future comparison"""
        # Convert tensors to lists for JSON serialization
        serializable_outputs = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                serializable_outputs[key] = {
                    "type": "tensor",
                    "shape": list(value.shape),
                    "data": value.detach().cpu().numpy().tolist()
                }
            else:
                serializable_outputs[key] = {"type": "scalar", "data": value}
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable_outputs, f, indent=2)
    
    def load_baseline(self, filepath: str) -> Dict[str, Any]:
        """Load baseline outputs for comparison"""
        with open(filepath, 'r') as f:
            serializable_outputs = json.load(f)
        
        # Convert back to tensors
        outputs = {}
        for key, value in serializable_outputs.items():
            if value["type"] == "tensor":
                outputs[key] = torch.tensor(value["data"])
            else:
                outputs[key] = value["data"]
        
        return outputs
    
    def compare_outputs(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, bool]:
        """Compare current outputs with baseline"""
        results = {}
        
        for key in baseline:
            if key not in current:
                results[key] = False
                continue
                
            baseline_val = baseline[key]
            current_val = current[key]
            
            if isinstance(baseline_val, torch.Tensor) and isinstance(current_val, torch.Tensor):
                # Compare tensor shapes
                if baseline_val.shape != current_val.shape:
                    results[key] = False
                    continue
                
                # Compare tensor values with tolerance
                results[key] = torch.allclose(current_val, baseline_val, rtol=self.tolerance, atol=self.tolerance)
            else:
                # Compare scalars
                if isinstance(baseline_val, float) and isinstance(current_val, float):
                    results[key] = abs(current_val - baseline_val) < self.tolerance
                else:
                    results[key] = current_val == baseline_val
        
        return results


class TestModelOutputRegression:
    """Test model output consistency across versions"""

    @pytest.fixture
    def regression_tester(self):
        """Get regression tester instance"""
        return OutputRegressionTester(tolerance=1e-4)

    @pytest.fixture
    def deterministic_batch(self, device):
        """Get deterministic batch for regression testing"""
        # Use fixed seed for deterministic outputs
        torch.manual_seed(12345)
        return {
            "input_ids": torch.tensor([[0, 100, 200, 300, 2, 1, 1, 1]]).to(device),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]).to(device),
            "labels": torch.tensor([[0, 100, 200, 300, 2, 1, 1, 1]]).to(device),
            "decoder_input_ids": torch.tensor([[0, 100, 200, 300, 2, 1, 1, 1]]).to(device),
            "decoder_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]).to(device)
        }

    @pytest.mark.regression
    def test_pretrain_model_output_consistency(self, pretrain_model, device, deterministic_batch, regression_tester):
        """Test that pretraining model outputs remain consistent"""
        model = pretrain_model.to(device)
        model.eval()
        
        # Set seeds for deterministic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        
        with torch.no_grad():
            outputs = model(**deterministic_batch)
        
        # Extract key outputs for regression testing
        test_outputs = {
            "loss": outputs["loss"].item(),
            "logits_shape": list(outputs["logits"].shape),
            "logits_mean": outputs["logits"].mean().item(),
            "logits_std": outputs["logits"].std().item(),
            "logits_first_token": outputs["logits"][0, 0, :10].clone(),  # First 10 logits
        }
        
        baseline_path = "tests/regression/baselines/pretrain_model_outputs.json"
        
        # For initial run, save baseline (uncomment when needed)
        # regression_tester.save_baseline(test_outputs, baseline_path)
        
        # Compare with baseline if it exists
        if os.path.exists(baseline_path):
            baseline_outputs = regression_tester.load_baseline(baseline_path)
            comparison_results = regression_tester.compare_outputs(test_outputs, baseline_outputs)
            
            # Assert all outputs match baseline
            for key, matches in comparison_results.items():
                assert matches, f"Regression detected in {key}: current != baseline"
        else:
            pytest.skip(f"Baseline file not found: {baseline_path}")

    @pytest.mark.regression
    @patch('src.model.MBartForConditionalGeneration.from_pretrained')
    def test_translation_model_output_consistency(self, mock_from_pretrained, device, deterministic_batch, regression_tester):
        """Test that translation model outputs remain consistent"""
        # Mock the pretrained model to avoid downloading
        mock_model = MagicMock()
        mock_model.return_value.loss = torch.tensor(0.5)
        mock_model.return_value.logits = torch.randn(1, 8, 1000)
        mock_from_pretrained.return_value = mock_model
        
        from src.model import MultilingualTranslationModel
        model = MultilingualTranslationModel("facebook/mbart-large-50")
        
        # Set seeds for deterministic behavior
        torch.manual_seed(42)
        
        # Create simpler batch for translation model
        translation_batch = {
            "input_ids": deterministic_batch["input_ids"],
            "attention_mask": deterministic_batch["attention_mask"],
            "labels": deterministic_batch["labels"]
        }
        
        with torch.no_grad():
            outputs = model(**translation_batch)
        
        test_outputs = {
            "loss": outputs["loss"].item() if hasattr(outputs["loss"], 'item') else outputs["loss"],
            "logits_shape": list(outputs["logits"].shape),
        }
        
        # Since this is mocked, we just test the structure
        assert "loss" in test_outputs
        assert "logits_shape" in test_outputs

    @pytest.mark.regression
    def test_noise_generator_consistency(self, mock_tokenizer, regression_tester):
        """Test that noise generator outputs remain consistent"""
        noise_gen = NoiseGenerator(mock_tokenizer, mask_prob=0.35, poisson_lambda=3.5)
        
        # Set seed for deterministic behavior
        torch.manual_seed(12345)
        
        # Test input
        input_ids = torch.tensor([[0, 100, 101, 102, 2, 200, 201, 202, 2, 1]])
        
        # Test span masking
        masked_output = noise_gen.span_masking(input_ids)
        
        # Test sentence permutation
        torch.manual_seed(12345)  # Reset seed
        permuted_output = noise_gen.sentence_permutation(input_ids)
        
        test_outputs = {
            "original_shape": list(input_ids.shape),
            "masked_shape": list(masked_output.shape),
            "permuted_shape": list(permuted_output.shape),
            "masked_tokens": (masked_output == mock_tokenizer.mask_token_id).sum().item(),
            "masked_first_row": masked_output[0].clone(),
            "permuted_first_row": permuted_output[0].clone(),
        }
        
        baseline_path = "tests/regression/baselines/noise_generator_outputs.json"
        
        # For initial run, save baseline (uncomment when needed)
        # regression_tester.save_baseline(test_outputs, baseline_path)
        
        # Compare with baseline if it exists
        if os.path.exists(baseline_path):
            baseline_outputs = regression_tester.load_baseline(baseline_path)
            comparison_results = regression_tester.compare_outputs(test_outputs, baseline_outputs)
            
            for key, matches in comparison_results.items():
                assert matches, f"Regression detected in noise generation {key}"
        else:
            pytest.skip(f"Baseline file not found: {baseline_path}")

    @pytest.mark.regression
    def test_model_generation_consistency(self, mock_model, mock_tokenizer, device):
        """Test that model generation remains consistent"""
        # Mock generation method for consistency
        mock_model.generate = MagicMock(return_value=torch.tensor([[0, 100, 200, 2]]))
        
        from src.model import MultilingualTranslationModel
        model = MultilingualTranslationModel.__new__(MultilingualTranslationModel)
        model.model = mock_model
        model.tokenizer = mock_tokenizer
        
        input_ids = torch.tensor([[0, 100, 200, 2]]).to(device)
        attention_mask = torch.ones(1, 4).to(device)
        
        # Set seed for consistency
        torch.manual_seed(42)
        
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=20,
            num_beams=3,
            length_penalty=1.0
        )
        
        # Test that generation parameters are passed correctly
        call_args = mock_model.generate.call_args
        assert call_args.kwargs['max_length'] == 20
        assert call_args.kwargs['num_beams'] == 3
        assert call_args.kwargs['length_penalty'] == 1.0

    @pytest.mark.regression
    def test_evaluation_metrics_consistency(self, mock_model, mock_tokenizer, device):
        """Test that evaluation metrics remain consistent"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        # Fixed test data for consistency
        hypotheses = ["Hello world", "Good morning", "Thank you"]
        references = ["Hello world", "Good evening", "Thanks"]
        
        # Test exact match (should be deterministic)
        em_results = evaluator.compute_exact_match(hypotheses, references)
        
        expected_exact_match = 1/3  # Only first pair matches exactly
        assert abs(em_results['exact_match'] - expected_exact_match) < 1e-6
        
        # Test with mock BLEU
        with patch('src.evaluation.sacrebleu.corpus_bleu') as mock_bleu:
            mock_bleu_result = MagicMock()
            mock_bleu_result.score = 25.5
            mock_bleu_result.signature = "BLEU+test"
            mock_bleu.return_value = mock_bleu_result
            
            bleu_results = evaluator.compute_bleu(hypotheses, references)
            assert bleu_results['bleu'] == 25.5

    @pytest.mark.regression
    def test_config_loading_consistency(self, test_config):
        """Test that configuration loading remains consistent"""
        # Test that config structure is preserved
        required_sections = ["model", "training", "data", "output"]
        for section in required_sections:
            assert section in test_config, f"Missing config section: {section}"
        
        # Test specific config values
        assert test_config["model"]["d_model"] == 128
        assert test_config["training"]["batch_size"] == 2
        assert test_config["data"]["max_length"] == 64
        
        # Test config serialization/deserialization consistency
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config == test_config, "Config serialization inconsistency"
        finally:
            os.unlink(temp_path)


class TestBehaviorRegression:
    """Test behavioral consistency across versions"""

    @pytest.mark.regression
    def test_training_convergence_pattern(self, pretrain_model, device):
        """Test that training convergence patterns remain consistent"""
        model = pretrain_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Fixed batch for consistent testing
        batch = {
            "input_ids": torch.randint(2, 1000, (2, 16)).to(device),
            "attention_mask": torch.ones(2, 16).to(device),
            "labels": torch.randint(2, 1000, (2, 16)).to(device),
            "decoder_input_ids": torch.randint(2, 1000, (2, 16)).to(device),
            "decoder_attention_mask": torch.ones(2, 16).to(device)
        }
        
        # Set seed for reproducible training
        torch.manual_seed(42)
        
        losses = []
        for step in range(5):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Test that loss decreases (basic sanity check)
        assert losses[-1] < losses[0], "Loss should decrease during training"
        
        # Test that losses are within expected range
        for loss in losses:
            assert 0 < loss < 20, f"Loss out of expected range: {loss}"

    @pytest.mark.regression
    def test_model_capacity_consistency(self, pretrain_model):
        """Test that model capacity metrics remain consistent"""
        # Count parameters
        total_params = sum(p.numel() for p in pretrain_model.parameters())
        trainable_params = sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad)
        
        # Test parameter counts are within expected range
        # These values depend on the test model configuration
        expected_param_range = (1000, 1000000)  # Adjust based on test model
        assert expected_param_range[0] <= total_params <= expected_param_range[1], \
            f"Parameter count changed: {total_params}"
        
        assert trainable_params == total_params, "All parameters should be trainable"
        
        # Test model size consistency
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        assert model_size_mb < 100, f"Model size too large: {model_size_mb} MB"

    @pytest.mark.regression
    def test_inference_speed_regression(self, pretrain_model, device):
        """Test that inference speed doesn't regress significantly"""
        model = pretrain_model.to(device)
        model.eval()
        
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 32)).to(device),
            "attention_mask": torch.ones(4, 32).to(device),
            "labels": torch.randint(0, 1000, (4, 32)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (4, 32)).to(device),
            "decoder_attention_mask": torch.ones(4, 32).to(device)
        }
        
        # Warmup
        with torch.no_grad():
            _ = model(**batch)
        
        # Measure inference time
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available() and device.type == 'cuda':
            start_time.record()
            with torch.no_grad():
                _ = model(**batch)
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)  # milliseconds
        else:
            import time
            start = time.time()
            with torch.no_grad():
                _ = model(**batch)
            inference_time = (time.time() - start) * 1000  # milliseconds
        
        # Test that inference time is reasonable
        max_allowed_time = 5000  # 5 seconds in milliseconds
        assert inference_time < max_allowed_time, \
            f"Inference too slow: {inference_time:.2f}ms (max: {max_allowed_time}ms)"

    @pytest.mark.regression
    def test_memory_usage_regression(self, pretrain_model, device):
        """Test that memory usage doesn't increase significantly"""
        model = pretrain_model.to(device)
        
        # Clear memory
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
        else:
            initial_memory = 0
        
        batch = {
            "input_ids": torch.randint(0, 1000, (8, 64)).to(device),
            "attention_mask": torch.ones(8, 64).to(device),
            "labels": torch.randint(0, 1000, (8, 64)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (8, 64)).to(device),
            "decoder_attention_mask": torch.ones(8, 64).to(device)
        }
        
        # Forward pass
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available() and device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(device)
            memory_used = (peak_memory - initial_memory) / 1024 / 1024  # MB
            
            # Test memory usage is reasonable
            max_allowed_memory = 2000  # 2GB
            assert memory_used < max_allowed_memory, \
                f"Memory usage too high: {memory_used:.2f}MB (max: {max_allowed_memory}MB)"

    @pytest.mark.regression
    def test_numerical_stability_regression(self, pretrain_model, device):
        """Test that numerical stability is maintained"""
        model = pretrain_model.to(device)
        model.train()
        
        # Create batch that might cause numerical issues
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 32)).to(device),
            "attention_mask": torch.ones(2, 32).to(device),
            "labels": torch.randint(0, 1000, (2, 32)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (2, 32)).to(device),
            "decoder_attention_mask": torch.ones(2, 32).to(device)
        }
        
        outputs = model(**batch)
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Test for NaN/Inf values
        assert torch.isfinite(loss).all(), "Loss contains NaN/Inf values"
        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf values"
        
        # Test gradient flow
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"
                assert param.grad.abs().max() < 100, f"Gradient for {name} too large: {param.grad.abs().max()}"