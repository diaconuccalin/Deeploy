# add_operator.py - Operator Addition Script

A standalone script to streamline adding new operators to Deeploy platforms.

## Overview

This script automates the boilerplate code generation for adding a new operator implementation to a specific platform with a specific data type. It generates all necessary files with clear TODOs, allowing you to focus on the operator-specific logic.

## Features

- **Automated File Generation**: Creates Parser, TypeChecker, Template, Binding, and Layer files
- **Smart Updates**: Appends to existing files or creates new ones
- **TODO Markers**: Clear placeholders for operator-specific implementation
- **SPDX Headers**: All files include proper license headers
- **Test Template**: Generates a test creation script
- **Dry-Run Mode**: Preview what will be generated without creating files
- **Input Validation**: Checks for valid platforms and data types

## Usage

### Basic Usage

```bash
cd /path/to/Deeploy
python scripts/add_operator.py --operator <OP_NAME> --platform <PLATFORM> --dtype <DTYPE>
```

### Examples

**Add FP32 Conv2D to Siracusa platform:**
```bash
python scripts/add_operator.py --operator Conv2D --platform Siracusa --dtype float32
```

**Add int8 MatMul to Generic platform:**
```bash
python scripts/add_operator.py -o MatMul -p Generic -d int8
```

**Add int16 Add to MemPool with interactive mode:**
```bash
python scripts/add_operator.py -o Add -p MemPool -d int16 --interactive
```

**Preview without creating files:**
```bash
python scripts/add_operator.py -o ReLU -p Snitch -d float16 --dry-run
```

## Generated Files

The script generates/updates the following files:

### 1. Template File
**Location:** `Deeploy/Targets/<PLATFORM>/Templates/<OPERATOR>Template.py`

Contains the Mako template for generating C code. Includes:
- Basic template structure
- Function signature placeholder
- Kernel implementation placeholder
- Optional tiled version

**TODO:** Implement the C kernel logic

### 2. Parser Class
**Location:** `Deeploy/Targets/<PLATFORM>/Parsers.py` (appended)

Parses ONNX node attributes. Includes:
- `parseNode()` method for attribute extraction
- `parseNodeCtxt()` method for context-aware parsing

**TODO:** Extract operator-specific attributes (kernel_shape, strides, etc.)

### 3. TypeChecker Class
**Location:** `Deeploy/Targets/<PLATFORM>/TypeCheckers.py` (appended)

Validates type compatibility. Includes:
- `typeCheck()` method with placeholder logic

**TODO:** Implement input/output type validation

### 4. Binding Definition
**Location:** `Deeploy/Targets/<PLATFORM>/Bindings.py` (appended)

Connects Parser, TypeChecker, and Template. Includes:
- Binding configuration
- Signature constraint placeholder

**TODO:** Update signature constraint based on operator requirements

### 5. Layer Class
**Location:** `Deeploy/Targets/<PLATFORM>/Layers.py` (appended)

Operator layer implementation. Includes:
- `computeOps()` method for profiling

**TODO:** Implement operation counting logic

### 6. Test Template Script
**Location:** `scripts/test_template_<OPERATOR>.py`

Helper script to create ONNX test network. Includes:
- Network creation function
- Input data generation
- Usage instructions

**TODO:** Define input/output shapes and operator attributes

## Workflow

### Step 1: Generate Boilerplate

```bash
python scripts/add_operator.py --operator MyOp --platform Siracusa --dtype float32
```

**Output:**
```
======================================================================
Generating MyOp operator for Siracusa platform (float32)
======================================================================

Created: Deeploy/Targets/Siracusa/Templates/MyOpTemplate.py
Updated: Deeploy/Targets/Siracusa/Parsers.py
Updated: Deeploy/Targets/Siracusa/TypeCheckers.py
Updated: Deeploy/Targets/Siracusa/Bindings.py
Updated: Deeploy/Targets/Siracusa/Layers.py
Created test template: scripts/test_template_MyOp.py

======================================================================
Manual step required: Update Deeploy/Targets/Siracusa/Platform.py
======================================================================

# Add to imports section:
from .Layers import MyOpLayer
from .Bindings import MyOpMapper

# Add to SiracusaMapping dictionary:
    'MyOp': MyOpLayer([MyOpMapper]),

======================================================================
```

### Step 2: Implement Operator Logic

Navigate to each generated file and fill in the TODOs:

**Parsers.py:**
```python
def parseNode(self, node) -> Dict:
    attrs = {}
    # TODO: Extract attributes
    if hasattr(node, 'attribute'):
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                attrs['kernel_shape'] = list(attr.ints)
    return attrs
```

**TypeCheckers.py:**
```python
def typeCheck(self, operatorRepresentation: OperatorRepresentation) -> bool:
    # TODO: Validate types
    if inputs[0].dtype != float32_t:
        return False
    return True
```

**Templates/MyOpTemplate.py:**
```python
myopTemplate = NodeTemplate("""
void ${name}(float *data_in, float *data_out, int size) {
    for (int i = 0; i < size; i++) {
        data_out[i] = compute(data_in[i]);
    }
}
""")
```

**Layers.py:**
```python
def computeOps(self, operatorRepresentation: OperatorRepresentation) -> int:
    # Calculate MACs/FLOPs
    return output_size * kernel_size
```

### Step 3: Update Platform Mapping

Edit `Deeploy/Targets/<PLATFORM>/Platform.py`:

```python
# Add imports
from .Layers import MyOpLayer
from .Bindings import MyOpMapper

# Update mapping
<PLATFORM>Mapping = {
    # ... existing mappings
    'MyOp': MyOpLayer([MyOpMapper]),
}
```

### Step 4: Create Test

```bash
# Generate test files
python scripts/test_template_MyOp.py

# This creates:
# - DeeployTest/Tests/MyOpTest/network.onnx
# - DeeployTest/Tests/MyOpTest/input.npy
```

### Step 5: Run Test

```bash
cd DeeployTest
python testRunner_siracusa.py -t Tests/MyOpTest --cores=8
```

### Step 6: Format and Lint

```bash
make format
make lint
```

## Parameters

### Required Arguments

- `-o, --operator`: Operator name (e.g., Conv2D, MatMul, Add)
- `-p, --platform`: Platform name (e.g., Generic, Siracusa, MemPool)
- `-d, --dtype`: Data type (e.g., int8, float32, float16)

### Optional Arguments

- `-i, --interactive`: Enable interactive mode for additional prompts
- `--dry-run`: Show what would be generated without creating files
- `-h, --help`: Show help message

## Supported Data Types

The script recognizes the following standard data types:

- **Integer:** `int8`, `int16`, `int32`, `uint8`, `uint16`, `uint32`
- **Floating-point:** `float16`, `float32`, `bfloat16`

Custom data types can be used, but you'll be prompted for confirmation.

## Supported Platforms

The script validates against existing platforms in `Deeploy/Targets/`:

- Generic
- CortexM
- MemPool
- Snitch
- Siracusa
- Neureka
- PULPOpen
- Chimera
- SoftHier

## Tips

### Adding Multiple Type Variants

To add the same operator with different data types:

```bash
# Add int8 version
python scripts/add_operator.py -o Conv2D -p Generic -d int8

# Add float32 version
python scripts/add_operator.py -o Conv2D -p Generic -d float32
```

Then update the Platform mapping to include both:

```python
Conv2DLayer([Conv2DMapperInt8, Conv2DMapperFloat32])
```

### Implementing Tiled Versions

For memory-constrained platforms, implement tiled versions:

```python
# In Template file
myopTemplateTiled = NodeTemplate("""
void ${name}_tiled(
    ${data_in.name}_t *${data_in.name},
    ${data_out.name}_t *${data_out.name},
    // DMA parameters
) {
    // Tiled implementation with DMA
}
""")
```

### Operator-Specific Attributes

Common attributes to extract in `parseNode()`:

**Conv2D:**
- `kernel_shape`: [kernel_h, kernel_w]
- `strides`: [stride_h, stride_w]
- `pads`: [pad_top, pad_left, pad_bottom, pad_right]
- `dilations`: [dilation_h, dilation_w]
- `group`: Number of groups

**MatMul:**
- `transA`: Transpose first input
- `transB`: Transpose second input

**Pooling:**
- `kernel_shape`: Pool window size
- `strides`: Pool stride
- `pads`: Padding

### Debugging

Use the Deeploy logger for debugging:

```python
from Deeploy.Logging import DEFAULT_LOGGER

DEFAULT_LOGGER.debug(f"Parsing {self.operator}: {attrs}")
```

## Limitations

1. **Manual Platform.py Update**: You must manually update the platform mapping
2. **Complex Operators**: Very complex operators may need additional manual work
3. **TileConstraints**: Tiling constraints must be added separately
4. **C Library**: You need to implement the actual C kernel in TargetLibraries/

## Troubleshooting

### "Platform does not exist"

Ensure you're using a valid platform name. Run with `--help` to see examples.

### "Class already exists"

The script detected an existing class. Choose to overwrite or manually merge.

### Import errors

Make sure to run the script from the repository root:
```bash
cd /path/to/Deeploy
python scripts/add_operator.py ...
```

## Contributing

When enhancing this script:
1. Maintain SPDX headers
2. Add comprehensive TODOs
3. Update this README
4. Test with dry-run mode
5. Follow Deeploy coding conventions

## Related Documentation

- Main documentation: `CLAUDE.md`
- Contribution guide: `CONTRIBUTING.md`
- Platform structure: `docs/structure.md`

## License

SPDX-License-Identifier: Apache-2.0
