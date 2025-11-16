# Deeploy Test Generation Guide

This guide explains how to create Deeploy test cases from PyTorch models using the automated test generation script.

## Overview

The `create_deeploy_test.py` script automates the process of creating Deeploy test cases from PyTorch models. It handles:

- ONNX export from PyTorch
- Random input generation
- Expected output computation
- Test artifact creation (network.onnx, inputs.npz, outputs.npz)
- Optional intermediate activation capture
- README generation

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch numpy onnx

# Optional: for ONNX graph simplification
pip install onnx-simplifier
```

### Basic Usage

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleGEMM \
    --test-name testSimpleGEMM \
    --input-shapes "1,32" \
    --seed 42 \
    --verbose
```

This will create a test directory at `DeeployTest/Tests/testSimpleGEMM/` containing:
- `network.onnx` - ONNX model
- `inputs.npz` - Test inputs
- `outputs.npz` - Expected outputs
- `README.md` - Test documentation

## Command-Line Arguments

### Required Arguments

- `--model-path PATH` - Path to Python file containing the model class
- `--model-class CLASS` - Name of the model class to instantiate
- `--test-name NAME` - Name of the test (directory name)
- `--input-shapes SHAPE [SHAPE ...]` - Input shapes as comma-separated values

### Optional Arguments

**Test Configuration:**
- `--output-dir DIR` - Parent directory for tests (default: `DeeployTest/Tests`)
- `--seed INT` - Random seed for reproducibility
- `--verbose` - Enable verbose output

**Input Generation:**
- `--input-range MIN MAX` - Range for random values (default: -1.0 1.0)

**Model Instantiation:**
- `--model-kwargs JSON` - JSON string of kwargs for model constructor

**ONNX Export:**
- `--opset-version INT` - ONNX opset version (default: 14)
- `--no-simplify` - Disable ONNX graph simplification

**Advanced:**
- `--capture-activations` - Capture intermediate activations (creates activations.npz)
- `--model-info TEXT` - Description for README

## Examples

### Example 1: Simple Linear Layer

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleLinear \
    --test-name testSimpleLinear \
    --input-shapes "1,10"
```

### Example 2: Convolutional Neural Network

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleCNN \
    --test-name testSimpleCNN \
    --input-shapes "1,3,32,32" \
    --seed 42
```

### Example 3: Model with Custom Parameters

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleLinear \
    --test-name testCustomLinear \
    --input-shapes "1,20" \
    --model-kwargs '{"input_dim": 20, "output_dim": 10}' \
    --seed 123
```

### Example 4: Multi-Input Model

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleElementwise \
    --test-name testElementwiseAdd \
    --input-shapes "1,16,32,32" "1,16,32,32" \
    --model-kwargs '{"operation": "add"}'
```

### Example 5: With Activations Capture

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleMLP \
    --test-name testMLPWithActivations \
    --input-shapes "1,784" \
    --capture-activations \
    --verbose
```

### Example 6: Custom Input Range

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleCNN \
    --test-name testCNNCustomRange \
    --input-shapes "1,3,32,32" \
    --input-range 0.0 1.0 \
    --seed 42
```

## Creating Custom Models

To use the test generator with your own models, create a Python file with your model class:

```python
# my_model.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

Then generate a test:

```bash
python create_deeploy_test.py \
    --model-path my_model.py \
    --model-class MyModel \
    --test-name testMyModel \
    --input-shapes "1,128"
```

## Running Generated Tests

After creating a test, run it on different platforms:

```bash
cd DeeployTest

# Generic platform (host machine)
python testRunner_generic.py -t Tests/testMyModel

# Cortex-M platform
python testRunner_cortexm.py -t Tests/testMyModel

# Siracusa platform
python testRunner_siracusa.py -t Tests/testMyModel --cores=8

# Siracusa with tiling
python testRunner_tiled_siracusa.py -t Tests/testMyModel --cores=8 --l1=16000 --l2=128000
```

## Best Practices

### 1. Use Reproducible Seeds

Always use `--seed` for reproducible tests:

```bash
--seed 42
```

### 2. Choose Appropriate Input Ranges

For models expecting normalized inputs (e.g., images), use:

```bash
--input-range 0.0 1.0
```

For models expecting standardized inputs, use:

```bash
--input-range -3.0 3.0
```

### 3. Test Small Models First

Start with simple models to verify the workflow:

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleGEMM \
    --test-name testGEMM \
    --input-shapes "1,32" \
    --seed 42
```

### 4. Use Verbose Mode for Debugging

Enable verbose output to see detailed information:

```bash
--verbose
```

### 5. Capture Activations for Debugging

For complex models, capture intermediate activations:

```bash
--capture-activations
```

### 6. Validate ONNX Export

After generation, visualize the ONNX model:

```bash
pip install netron
netron DeeployTest/Tests/testMyModel/network.onnx
```

## Workflow Integration

### Recommended Workflow

1. **Design Model** - Create PyTorch model in a separate file
2. **Test PyTorch Model** - Verify model works with PyTorch
3. **Generate Test** - Use `create_deeploy_test.py`
4. **Verify ONNX** - Check ONNX model with Netron
5. **Run on Generic** - Test on generic platform first
6. **Run on Target** - Test on target platform
7. **Optimize** - Add tiling, quantization, etc.

### Example Workflow

```bash
# Step 1 & 2: Create and test PyTorch model
python my_model.py  # Should include test code

# Step 3: Generate Deeploy test
python create_deeploy_test.py \
    --model-path my_model.py \
    --model-class MyModel \
    --test-name testMyModel \
    --input-shapes "1,128" \
    --seed 42 \
    --verbose

# Step 4: Visualize ONNX
netron DeeployTest/Tests/testMyModel/network.onnx

# Step 5: Test on generic platform
cd DeeployTest
python testRunner_generic.py -t Tests/testMyModel -vv

# Step 6: Test on target platform
python testRunner_siracusa.py -t Tests/testMyModel --cores=8

# Step 7: Test with tiling
python testRunner_tiled_siracusa.py -t Tests/testMyModel \
    --cores=8 --l1=16000 --l2=128000
```

## Troubleshooting

### Issue: ONNX Export Fails

**Solution**: Some PyTorch operations aren't supported by ONNX. Try:
- Using different opset version: `--opset-version 15`
- Simplifying model architecture
- Checking PyTorch ONNX compatibility docs

### Issue: Shape Mismatch

**Solution**: Ensure input shapes match model expectations:
- Check model's expected input dimensions
- Verify batch dimension (usually first dimension)
- Use correct number of inputs for multi-input models

### Issue: Import Errors

**Solution**: Ensure all dependencies are installed:
```bash
pip install torch numpy onnx onnx-simplifier
```

### Issue: Test Fails on Platform

**Solution**:
- First test on generic platform
- Check generated C code for errors
- Verify platform-specific constraints (memory, operations)
- Use `-vvv` for detailed logs

## Advanced Usage

### Custom Preprocessing

For models requiring specific preprocessing:

```python
# custom_model.py
class PreprocessedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def forward(self, x):
        # Apply preprocessing
        x = (x - 0.5) / 0.5  # Normalize to [-1, 1]
        return self.model(x)
```

### Multi-Output Models

The script handles multi-output models automatically:

```python
class MultiOutputModel(nn.Module):
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1, out2  # Multiple outputs
```

### Stateful Models (RNN, LSTM)

For recurrent models, specify sequence dimensions:

```bash
python create_deeploy_test.py \
    --model-path example_models.py \
    --model-class SimpleRNN \
    --test-name testRNN \
    --input-shapes "1,10,20"  # (batch, seq_len, input_dim)
```

## Integration with CI/CD

Add test generation to your CI/CD pipeline:

```yaml
# .github/workflows/generate-tests.yml
name: Generate Tests

on: [push]

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install torch numpy onnx
      - name: Generate test
        run: |
          python create_deeploy_test.py \
            --model-path models/my_model.py \
            --model-class MyModel \
            --test-name testMyModel \
            --input-shapes "1,128" \
            --seed 42
      - name: Run test
        run: |
          cd DeeployTest
          python testRunner_generic.py -t Tests/testMyModel
```

## See Also

- [CLAUDE.md](CLAUDE.md) - AI assistant guide for Deeploy
- [Deeploy Documentation](https://pulp-platform.github.io/Deeploy/)
- [ONNX Documentation](https://onnx.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## Contributing

When adding new example models to `example_models.py`:

1. Follow the naming convention: `Simple*`
2. Add docstring with input/output shapes
3. Test the model in the `__main__` block
4. Add example usage in docstring

## License

This tool is part of Deeploy and follows the same license (Apache 2.0).
