"""
Performance benchmark tests
"""

import pytest
import torch
import time
import psutil
import gc
from unittest.mock import patch, MagicMock
import numpy as np
from typing import Dict, List

from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator
from src.data import DataProcessor, TranslationDataset
from src.trainer import DenoisingPretrainingTrainer, TranslationTrainer
from src.evaluation import TranslationEvaluator


class PerformanceProfiler:
    """Utility class for performance profiling"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        
    def start(self):
        """Start profiling"""
        gc.collect()  # Clean up before measurement
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update_peak_memory(self):
        """Update peak memory usage"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def stop(self) -> Dict[str, float]:
        """Stop profiling and return metrics"""
        self.update_peak_memory()
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            "duration_seconds": end_time - self.start_time,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": end_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": end_memory - self.start_memory
        }


class TestModelPerformance:
    """Test model performance characteristics"""

    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_model_inference_speed(self, pretrain_model, device, batch_size):
        """Test model inference speed with different batch sizes"""
        model = pretrain_model.to(device)
        model.eval()
        
        # Create batch
        seq_len = 64
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "attention_mask": torch.ones(batch_size, seq_len).to(device),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_attention_mask": torch.ones(batch_size, seq_len).to(device)
        }
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(**batch)
        
        # Benchmark
        profiler = PerformanceProfiler()
        profiler.start()
        
        num_runs = 10
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(**batch)
                profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        # Performance assertions
        avg_time_per_batch = metrics["duration_seconds"] / num_runs
        samples_per_second = batch_size / avg_time_per_batch
        
        print(f"Batch size: {batch_size}")
        print(f"Avg time per batch: {avg_time_per_batch:.4f}s")
        print(f"Samples per second: {samples_per_second:.2f}")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        # Basic performance checks
        assert avg_time_per_batch < 5.0, f"Inference too slow: {avg_time_per_batch}s per batch"
        assert metrics['peak_memory_mb'] < 8000, f"Memory usage too high: {metrics['peak_memory_mb']} MB"

    @pytest.mark.performance
    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256])
    def test_model_sequence_length_scaling(self, pretrain_model, device, seq_len):
        """Test how model performance scales with sequence length"""
        model = pretrain_model.to(device)
        model.eval()
        
        batch_size = 2
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "attention_mask": torch.ones(batch_size, seq_len).to(device),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_attention_mask": torch.ones(batch_size, seq_len).to(device)
        }
        
        # Warmup
        with torch.no_grad():
            _ = model(**batch)
        
        # Benchmark
        profiler = PerformanceProfiler()
        profiler.start()
        
        with torch.no_grad():
            output = model(**batch)
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        print(f"Sequence length: {seq_len}")
        print(f"Duration: {metrics['duration_seconds']:.4f}s")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        # Performance should scale reasonably with sequence length
        expected_max_time = seq_len * 0.01  # Rough heuristic
        assert metrics['duration_seconds'] < expected_max_time, \
            f"Too slow for seq_len {seq_len}: {metrics['duration_seconds']}s"

    @pytest.mark.performance
    def test_memory_usage_patterns(self, pretrain_model, device):
        """Test memory usage patterns during training"""
        model = pretrain_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 32)).to(device),
            "attention_mask": torch.ones(4, 32).to(device),
            "labels": torch.randint(0, 1000, (4, 32)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (4, 32)).to(device),
            "decoder_attention_mask": torch.ones(4, 32).to(device)
        }
        
        memory_usage = []
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Simulate training steps
        for step in range(5):
            model.train()
            optimizer.zero_grad()
            
            output = model(**batch)
            loss = output['loss']
            loss.backward()
            
            profiler.update_peak_memory()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
            
            optimizer.step()
        
        metrics = profiler.stop()
        
        # Check memory usage patterns
        memory_increase = max(memory_usage) - min(memory_usage)
        print(f"Memory increase during training: {memory_increase:.2f} MB")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        # Memory should not grow unboundedly
        assert memory_increase < 1000, f"Memory increase too large: {memory_increase} MB"

    @pytest.mark.performance
    def test_gpu_memory_efficiency(self, pretrain_model):
        """Test GPU memory efficiency"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda")
        model = pretrain_model.to(device)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        batch_size = 8
        seq_len = 128
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "attention_mask": torch.ones(batch_size, seq_len).to(device),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "decoder_attention_mask": torch.ones(batch_size, seq_len).to(device)
        }
        
        # Forward pass
        with torch.no_grad():
            output = model(**batch)
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        current_memory = torch.cuda.memory_allocated(device)
        
        memory_mb = (peak_memory - initial_memory) / 1024 / 1024
        
        print(f"GPU memory used: {memory_mb:.2f} MB")
        print(f"Memory per sample: {memory_mb / batch_size:.2f} MB")
        
        # Check reasonable GPU memory usage
        assert memory_mb < 2000, f"GPU memory usage too high: {memory_mb} MB"

    @pytest.mark.performance
    def test_model_parameter_count(self, pretrain_model):
        """Test that model parameter count is reasonable"""
        total_params = sum(p.numel() for p in pretrain_model.parameters())
        trainable_params = sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad)
        
        param_size_mb = total_params * 4 / 1024 / 1024  # Assuming float32
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {param_size_mb:.2f} MB")
        
        # Check reasonable parameter count for test model
        assert total_params < 50_000_000, f"Too many parameters: {total_params:,}"
        assert trainable_params == total_params, "All parameters should be trainable"


class TestDataPerformance:
    """Test data processing performance"""

    @pytest.mark.performance
    def test_dataset_iteration_speed(self, sample_translation_data, mock_tokenizer):
        """Test dataset iteration performance"""
        # Create larger dataset for meaningful benchmarking
        large_data = sample_translation_data * 200  # 1000 items
        dataset = TranslationDataset(large_data, mock_tokenizer, max_length=64)
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Iterate through dataset
        items_processed = 0
        for i in range(min(100, len(dataset))):  # Process subset for speed
            item = dataset[i]
            items_processed += 1
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        items_per_second = items_processed / metrics["duration_seconds"]
        
        print(f"Items processed: {items_processed}")
        print(f"Items per second: {items_per_second:.2f}")
        print(f"Memory usage: {metrics['peak_memory_mb']:.2f} MB")
        
        # Performance expectations
        assert items_per_second > 50, f"Dataset iteration too slow: {items_per_second:.2f} items/s"

    @pytest.mark.performance
    def test_dataloader_performance(self, sample_translation_data, mock_tokenizer):
        """Test DataLoader performance with multiple workers"""
        from torch.utils.data import DataLoader
        
        large_data = sample_translation_data * 100
        dataset = TranslationDataset(large_data, mock_tokenizer)
        
        # Test different worker configurations
        for num_workers in [0, 1, 2]:
            dataloader = DataLoader(
                dataset, 
                batch_size=16, 
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            profiler = PerformanceProfiler()
            profiler.start()
            
            batches_processed = 0
            for batch in dataloader:
                batches_processed += 1
                profiler.update_peak_memory()
                if batches_processed >= 10:  # Limit for testing
                    break
            
            metrics = profiler.stop()
            
            batches_per_second = batches_processed / metrics["duration_seconds"]
            
            print(f"Workers: {num_workers}, Batches/s: {batches_per_second:.2f}")
            
            # Basic performance check
            assert batches_per_second > 0.1, f"DataLoader too slow with {num_workers} workers"

    @pytest.mark.performance
    @patch('src.data.load_dataset')
    def test_data_loading_performance(self, mock_load_dataset):
        """Test data loading performance"""
        # Mock large dataset
        large_mock_data = [
            {"translation": {"en": f"Source {i}", "ro": f"Target {i}"}}
            for i in range(10000)
        ]
        mock_load_dataset.return_value = large_mock_data
        
        processor = DataProcessor.__new__(DataProcessor)
        processor.tokenizer = MagicMock()
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Load data
        data = processor.load_wmt_en_ro("train")
        profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        items_per_second = len(data) / metrics["duration_seconds"]
        
        print(f"Loaded {len(data)} items in {metrics['duration_seconds']:.2f}s")
        print(f"Loading speed: {items_per_second:.2f} items/s")
        
        assert items_per_second > 1000, f"Data loading too slow: {items_per_second:.2f} items/s"

    @pytest.mark.performance
    def test_tokenization_performance(self, mock_tokenizer):
        """Test tokenization performance"""
        # Create test texts
        test_texts = [f"This is test sentence number {i}." for i in range(1000)]
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Tokenize texts
        for text in test_texts:
            tokens = mock_tokenizer(text, return_tensors="pt", max_length=64, 
                                  padding="max_length", truncation=True)
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        texts_per_second = len(test_texts) / metrics["duration_seconds"]
        
        print(f"Tokenized {len(test_texts)} texts in {metrics['duration_seconds']:.2f}s")
        print(f"Tokenization speed: {texts_per_second:.2f} texts/s")
        
        assert texts_per_second > 100, f"Tokenization too slow: {texts_per_second:.2f} texts/s"


class TestTrainingPerformance:
    """Test training performance"""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_training_step_performance(self, pretrain_model, device, test_config, temp_checkpoint_dir):
        """Test single training step performance"""
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        # Mock batch
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 32)).to(device),
            "attention_mask": torch.ones(4, 32).to(device),
            "labels": torch.randint(0, 1000, (4, 32)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (4, 32)).to(device),
            "decoder_attention_mask": torch.ones(4, 32).to(device)
        }
        
        trainer = DenoisingPretrainingTrainer(
            model=pretrain_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        # Warmup
        trainer.train_step(batch)
        
        # Benchmark training steps
        profiler = PerformanceProfiler()
        profiler.start()
        
        num_steps = 10
        for _ in range(num_steps):
            loss = trainer.train_step(batch)
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        steps_per_second = num_steps / metrics["duration_seconds"]
        
        print(f"Training steps per second: {steps_per_second:.2f}")
        print(f"Average loss: {loss:.4f}")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        assert steps_per_second > 0.5, f"Training too slow: {steps_per_second:.2f} steps/s"

    @pytest.mark.performance
    def test_validation_performance(self, pretrain_model, device, test_config, temp_checkpoint_dir):
        """Test validation performance"""
        train_loader = MagicMock()
        
        # Mock validation data
        val_batches = [
            {
                "input_ids": torch.randint(0, 1000, (2, 32)).to(device),
                "attention_mask": torch.ones(2, 32).to(device),
                "labels": torch.randint(0, 1000, (2, 32)).to(device),
                "decoder_input_ids": torch.randint(0, 1000, (2, 32)).to(device),
                "decoder_attention_mask": torch.ones(2, 32).to(device)
            }
            for _ in range(5)
        ]
        
        val_loader = MagicMock()
        val_loader.__iter__ = MagicMock(return_value=iter(val_batches))
        
        trainer = DenoisingPretrainingTrainer(
            model=pretrain_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=temp_checkpoint_dir,
            config=test_config
        )
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        val_loss = trainer.validate()
        
        metrics = profiler.stop()
        
        batches_per_second = len(val_batches) / metrics["duration_seconds"]
        
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation batches per second: {batches_per_second:.2f}")
        print(f"Duration: {metrics['duration_seconds']:.2f}s")
        
        assert batches_per_second > 1.0, f"Validation too slow: {batches_per_second:.2f} batches/s"

    @pytest.mark.performance
    def test_checkpoint_save_performance(self, mock_model, device, temp_checkpoint_dir):
        """Test checkpoint saving performance"""
        from src.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=mock_model,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            device=device,
            output_dir=temp_checkpoint_dir
        )
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Save multiple checkpoints
        for i in range(3):
            trainer.save_checkpoint(f"test_checkpoint_{i}.pt")
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        saves_per_second = 3 / metrics["duration_seconds"]
        
        print(f"Checkpoint saves per second: {saves_per_second:.2f}")
        print(f"Duration: {metrics['duration_seconds']:.2f}s")
        
        assert saves_per_second > 0.1, f"Checkpoint saving too slow: {saves_per_second:.2f} saves/s"


class TestEvaluationPerformance:
    """Test evaluation performance"""

    @pytest.mark.performance
    def test_translation_performance(self, mock_model, mock_tokenizer, device):
        """Test translation speed performance"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        # Create test sentences
        test_sentences = [f"Test sentence number {i}." for i in range(100)]
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        translations = evaluator.translate_batch(test_sentences, batch_size=8)
        profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        sentences_per_second = len(test_sentences) / metrics["duration_seconds"]
        
        print(f"Translation speed: {sentences_per_second:.2f} sentences/s")
        print(f"Duration: {metrics['duration_seconds']:.2f}s")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        assert sentences_per_second > 10, f"Translation too slow: {sentences_per_second:.2f} sentences/s"

    @pytest.mark.performance
    def test_metric_computation_performance(self, mock_model, mock_tokenizer, device):
        """Test evaluation metric computation performance"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        # Create test data
        hypotheses = [f"Hypothesis {i}" for i in range(1000)]
        references = [f"Reference {i}" for i in range(1000)]
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Test different metrics
        em_results = evaluator.compute_exact_match(hypotheses, references)
        profiler.update_peak_memory()
        
        with patch('src.evaluation.sacrebleu.corpus_bleu') as mock_bleu:
            mock_bleu.return_value = MagicMock(score=25.0, signature="test")
            bleu_results = evaluator.compute_bleu(hypotheses, references)
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        pairs_per_second = len(hypotheses) / metrics["duration_seconds"]
        
        print(f"Metric computation speed: {pairs_per_second:.2f} pairs/s")
        print(f"Duration: {metrics['duration_seconds']:.2f}s")
        
        assert pairs_per_second > 100, f"Metric computation too slow: {pairs_per_second:.2f} pairs/s"

    @pytest.mark.performance
    def test_large_evaluation_dataset(self, mock_model, mock_tokenizer, device):
        """Test evaluation on large dataset"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        # Create large evaluation dataset
        source_texts = [f"Source sentence {i}" for i in range(500)]
        target_texts = [f"Target sentence {i}" for i in range(500)]
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        # Mock translation to focus on evaluation overhead
        with patch.object(evaluator, 'translate_batch') as mock_translate:
            mock_translate.return_value = [f"Translation {i}" for i in range(500)]
            
            results, hyps, refs = evaluator.evaluate_dataset(
                source_texts, target_texts, batch_size=32
            )
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        samples_per_second = len(source_texts) / metrics["duration_seconds"]
        
        print(f"Evaluation speed: {samples_per_second:.2f} samples/s")
        print(f"Duration: {metrics['duration_seconds']:.2f}s")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        assert samples_per_second > 50, f"Evaluation too slow: {samples_per_second:.2f} samples/s"


class TestScalabilityLimits:
    """Test scalability limits and bottlenecks"""

    @pytest.mark.performance
    @pytest.mark.parametrize("model_size", ["small", "medium"])
    def test_model_size_scaling(self, model_size, device):
        """Test performance scaling with model size"""
        from transformers import MBartConfig
        from src.model import MultilingualDenoisingPretraining
        
        if model_size == "small":
            config = MBartConfig(
                vocab_size=1000, d_model=256, encoder_layers=2, decoder_layers=2,
                encoder_attention_heads=4, decoder_attention_heads=4,
                encoder_ffn_dim=512, decoder_ffn_dim=512
            )
        else:  # medium
            config = MBartConfig(
                vocab_size=1000, d_model=512, encoder_layers=4, decoder_layers=4,
                encoder_attention_heads=8, decoder_attention_heads=8,
                encoder_ffn_dim=1024, decoder_ffn_dim=1024
            )
        
        model = MultilingualDenoisingPretraining(config).to(device)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 32)).to(device),
            "attention_mask": torch.ones(2, 32).to(device),
            "labels": torch.randint(0, 1000, (2, 32)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (2, 32)).to(device),
            "decoder_attention_mask": torch.ones(2, 32).to(device)
        }
        
        profiler = PerformanceProfiler()
        profiler.start()
        
        with torch.no_grad():
            output = model(**batch)
            profiler.update_peak_memory()
        
        metrics = profiler.stop()
        
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"Model size: {model_size}")
        print(f"Parameters: {param_count:,}")
        print(f"Duration: {metrics['duration_seconds']:.4f}s")
        print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
        
        # Performance should be reasonable for both sizes
        assert metrics['duration_seconds'] < 2.0, f"{model_size} model too slow"

    @pytest.mark.performance
    def test_memory_leak_detection(self, pretrain_model, device):
        """Test for memory leaks during repeated operations"""
        model = pretrain_model.to(device)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 32)).to(device),
            "attention_mask": torch.ones(2, 32).to(device),
            "labels": torch.randint(0, 1000, (2, 32)).to(device),
            "decoder_input_ids": torch.randint(0, 1000, (2, 32)).to(device),
            "decoder_attention_mask": torch.ones(2, 32).to(device)
        }
        
        memory_usage = []
        
        for i in range(10):
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Simulate training step
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            output = model(**batch)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(end_memory)
            
            del optimizer
        
        # Check for memory leaks
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        
        print(f"Memory usage trend: {memory_trend:.2f} MB/iteration")
        print(f"Memory usage: {memory_usage}")
        
        # Memory should not consistently increase
        assert memory_trend < 5.0, f"Potential memory leak: {memory_trend:.2f} MB/iteration"