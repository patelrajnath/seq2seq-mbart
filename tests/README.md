# mBART Seq-to-Seq Testing Infrastructure

This directory contains a comprehensive test suite for the mBART sequence-to-sequence translation project. The testing infrastructure is designed to ensure code quality, reliability, and performance across all components.

## 📁 Directory Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                   # Shared pytest fixtures and configuration
├── unit/                         # Unit tests for individual components
│   ├── test_model_components.py     # Model classes (MultilingualDenoisingPretraining, etc.)
│   ├── test_data_components.py      # Data processing (DataProcessor, datasets)
│   ├── test_trainer_components.py   # Training components (trainers, optimizers)
│   └── test_evaluation_components.py # Evaluation metrics and utilities
├── integration/                  # Integration tests for complete workflows
│   ├── test_training_pipeline.py    # End-to-end training workflows
│   └── test_config_management.py    # Configuration loading and validation
├── performance/                  # Performance and benchmarking tests
│   └── test_performance_benchmarks.py # Speed, memory, scalability tests
├── regression/                   # Regression tests for consistency
│   └── test_model_outputs.py       # Model output consistency across versions
├── error_handling/              # Robustness and error handling tests
│   ├── test_robustness.py          # Error conditions and recovery
│   └── test_edge_cases.py          # Edge cases and boundary conditions
├── fixtures/                    # Test data and utilities
│   └── test_data_fixtures.py      # Synthetic data generation and fixtures
├── data/                        # Generated test datasets
└── README.md                    # This file
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation:

- **Model Components**: Test model classes, noise generators, and neural network layers
- **Data Components**: Test dataset classes, data processors, and tokenization
- **Trainer Components**: Test training loops, optimizers, and schedulers
- **Evaluation Components**: Test metrics computation and evaluation utilities

**Run unit tests:**
```bash
python run_tests.py --type unit
```

### Integration Tests (`tests/integration/`)
Test complete workflows and component interactions:

- **Training Pipeline**: End-to-end pretraining → finetuning → evaluation
- **Configuration Management**: Config loading, validation, and parameter overrides

**Run integration tests:**
```bash
python run_tests.py --type integration
```

### Performance Tests (`tests/performance/`)
Benchmark performance characteristics:

- **Model Performance**: Inference speed, memory usage, scalability
- **Data Performance**: Loading speed, tokenization performance
- **Training Performance**: Training throughput, convergence speed

**Run performance tests:**
```bash
python run_tests.py --type performance
```

### Regression Tests (`tests/regression/`)
Ensure consistency across code changes:

- **Model Outputs**: Verify model outputs remain consistent
- **Behavior Patterns**: Test training convergence and numerical stability

**Run regression tests:**
```bash
python run_tests.py --type regression
```

### Error Handling Tests (`tests/error_handling/`)
Test robustness under adverse conditions:

- **Robustness**: Memory exhaustion, corrupted data, network failures
- **Edge Cases**: Empty inputs, extreme values, malformed data

**Run error handling tests:**
```bash
python run_tests.py --type error_handling
```

## 🚀 Quick Start

### Prerequisites

Install required testing dependencies:
```bash
pip install pytest pytest-cov pytest-mock psutil numpy
```

### Running Tests

**Run all tests:**
```bash
python run_tests.py
```

**Run specific test types:**
```bash
python run_tests.py --type unit          # Unit tests only
python run_tests.py --type integration   # Integration tests only
python run_tests.py --type performance   # Performance tests only
```

**Run with coverage:**
```bash
python run_tests.py --coverage
```

**Fast testing (skip slow tests):**
```bash
python run_tests.py --fast
```

**Parallel execution:**
```bash
python run_tests.py --parallel 4
```

### Direct pytest Usage

You can also run tests directly with pytest:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/test_model_components.py

# Specific test function
pytest tests/unit/test_model_components.py::TestMultilingualDenoisingPretraining::test_initialization

# With markers
pytest -m "unit and not slow"
pytest -m "integration or performance"

# With coverage
pytest --cov=src --cov-report=html tests/
```

## 🏷️ Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.regression` - Regression tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.network` - Tests requiring network access

**Filter tests by markers:**
```bash
pytest -m "unit and not slow"           # Fast unit tests only
pytest -m "performance or regression"   # Performance and regression tests
pytest -m "not gpu"                     # Skip GPU tests
```

## 📊 Coverage Reporting

Generate coverage reports:

```bash
# Terminal coverage report
pytest --cov=src --cov-report=term-missing tests/

# HTML coverage report
pytest --cov=src --cov-report=html tests/
# View report: open htmlcov/index.html
```

## 🔧 Configuration

### pytest.ini
Main pytest configuration including:
- Test discovery patterns
- Coverage settings
- Markers definition
- Warning filters

### conftest.py
Shared fixtures and utilities:
- Device fixtures (CPU/GPU)
- Model fixtures (small test models)
- Data fixtures (sample datasets)
- Mock objects (tokenizers, models)
- Temporary directories and files

## 📈 Test Data Generation

The `fixtures/test_data_fixtures.py` module provides:

- **SyntheticDataGenerator**: Creates realistic test data
- **TestDataFixtures**: Predefined test configurations
- **create_comprehensive_test_data()**: Generates complete test datasets

Generate test data:
```bash
python tests/fixtures/test_data_fixtures.py
```

## 🎯 Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_model_handles_empty_input_gracefully`
2. **Test one thing per test**: Focus on specific functionality
3. **Use fixtures**: Leverage shared setup via conftest.py
4. **Mark appropriately**: Add relevant pytest markers
5. **Mock external dependencies**: Avoid network calls, file I/O in unit tests

### Test Organization

1. **Group related tests**: Use test classes for logical grouping
2. **Parametrize tests**: Test multiple scenarios efficiently
3. **Use appropriate test types**: Unit vs integration vs performance
4. **Document complex tests**: Add docstrings explaining test purpose

### Performance Considerations

1. **Skip slow tests in development**: Use `--fast` flag
2. **Run performance tests separately**: They require more time
3. **Use smaller models for testing**: Avoid large pretrained models
4. **Clean up resources**: Prevent memory leaks in tests

## 🚨 Continuous Integration

For CI/CD pipelines, use these commands:

```bash
# Fast test suite for pull requests
python run_tests.py --type unit --fast --coverage

# Full test suite for main branch
python run_tests.py --coverage

# Performance regression testing
python run_tests.py --type performance --type regression
```

## 🛠️ Troubleshooting

### Common Issues

**CUDA/GPU Tests Failing:**
```bash
# Skip GPU tests if CUDA not available
python run_tests.py --type unit -m "not gpu"
```

**Memory Issues:**
```bash
# Run tests with smaller batch sizes
export PYTEST_CURRENT_TEST_BATCH_SIZE=1
python run_tests.py --fast
```

**Import Errors:**
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python run_tests.py
```

**Slow Tests:**
```bash
# Run only fast tests during development
python run_tests.py --fast --type unit
```

### Debug Mode

Run tests with more detailed output:
```bash
python run_tests.py --verbose
pytest -vvv tests/unit/test_model_components.py::test_specific_function
```

## 📋 Test Coverage Goals

- **Unit Tests**: >90% line coverage for all src/ modules
- **Integration Tests**: All major workflows covered
- **Error Handling**: All error conditions tested
- **Performance**: Benchmarks for all critical paths
- **Regression**: Key model behaviors protected

## 🤝 Contributing

When adding new features:

1. **Write tests first** (TDD approach recommended)
2. **Add appropriate markers** for test categorization
3. **Update fixtures** if new test data patterns needed
4. **Document complex test logic**
5. **Ensure tests pass locally** before submitting

### Adding New Tests

1. Choose appropriate test category (`unit/`, `integration/`, etc.)
2. Use existing fixtures from `conftest.py`
3. Follow naming conventions: `test_<functionality>_<scenario>`
4. Add docstrings explaining test purpose
5. Use appropriate assertions and error messages

## 📚 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Coverage Plugin](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Fixture Documentation](https://docs.pytest.org/en/stable/fixture.html)

---

For questions about the testing infrastructure, please check the existing tests for examples or consult the project documentation.