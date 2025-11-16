# CLAUDE.md - Deeploy Tiling System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Tiling Constraints Architecture](#tiling-constraints-architecture)
3. [Understanding TileConstraints](#understanding-tileconstraints)
4. [How to Write Custom TileConstraints](#how-to-write-custom-tileconstraints)
5. [Debugging Tiling Constraints](#debugging-tiling-constraints)
6. [Common Patterns and Examples](#common-patterns-and-examples)
7. [Target-Specific TileConstraints](#target-specific-tileconstraints)

---

## Overview

### What is Tiling?

Tiling is a memory optimization technique used in Deeploy to partition large neural network operations into smaller chunks (tiles) that fit within the limited on-chip memory of embedded devices. Instead of loading entire tensors into memory, the framework:

1. **Divides** large tensors into smaller hyperrectangular tiles
2. **Loads** only the required tiles into fast on-chip memory (L1)
3. **Processes** each tile independently
4. **Stores** results back to slower memory (L2/L3)

### Why TileConstraints?

TileConstraints define the **rules and relationships** that govern how tensors can be tiled. They ensure:

- **Correctness**: Tiles maintain mathematical correctness of operations
- **Feasibility**: Tiles fit within hardware memory constraints
- **Efficiency**: Tiling strategy optimizes for performance

---

## Tiling Constraints Architecture

### Core Components

```
Deeploy/TilingExtension/
├── TileConstraint.py          # Base class for all tile constraints
├── TilerModel.py              # OR-Tools constraint solver model
├── TilingCodegen.py           # Code generation from tiling solutions
├── MemoryConstraints.py       # Memory allocation constraints
└── MemoryScheduler.py         # Memory scheduling algorithms
```

### Target-Specific Constraints

```
Deeploy/Targets/[target_name]/TileConstraints/
├── __init__.py
├── ConvTileConstraint.py      # Example: Convolution tiling
├── GemmTileConstraint.py      # Example: Matrix multiplication tiling
└── ...                        # Other operator-specific constraints
```

### The TileConstraint Base Class

Located in: `Deeploy/TilingExtension/TileConstraint.py`

**Key Methods (Override these):**

| Method | Purpose | Required |
|--------|---------|----------|
| `addGeometricalConstraint()` | Define mathematical relationships between input/output tensor dimensions | Yes |
| `addPolicyConstraint()` | Add platform-specific constraints (alignment, minimum sizes, etc.) | Optional |
| `serializeTilingSolution()` | Convert solver solution into memory load/store schedules | Yes |
| `constructSymbolicNodeRep()` | Create symbolic representation for template variables | Optional |

---

## Understanding TileConstraints

### 1. Geometrical Constraints

**Purpose**: Define how output dimensions relate to input dimensions

**Example** (from Conv2D):
```python
@staticmethod
def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict,
                             ctxt: NetworkContext) -> TilerModel:
    # Extract tensor dimensions as constraint variables
    inputHeightVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=1)
    outputHeightVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=1)

    # Define relationship: output_height = (input_height - kernel_height) / stride + 1
    tilerModel.addConstraint(
        outputHeightVar == (inputHeightVar - kernelHeight) // stride + 1
    )

    return tilerModel
```

**Key Concepts**:
- Variables represent **tile dimensions**, not full tensor dimensions
- Constraints are **mathematical equations** using OR-Tools expressions
- Must account for **padding, strides, dilations, etc.**

### 2. Policy Constraints

**Purpose**: Enforce platform-specific requirements and optimizations

**Example** (from Conv2D):
```python
@staticmethod
def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict,
                       ctxt: NetworkContext) -> TilerModel:
    # Keep entire input channels (required for im2col algorithm)
    tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])

    # Minimum spatial dimensions must be at least kernel size
    tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'])

    # Input tiles must be compatible with stride
    tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)

    # Keep entire weight dimensions
    tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])

    return tilerModel
```

**Common Policy Patterns**:
- **No tiling on certain dimensions**: `var == full_size`
- **Minimum tile sizes**: `var >= min_value`
- **Alignment constraints**: `(var % modulo) == 0`
- **Performance hints**: `tilerModel.addConstraint(..., strategy=PerformanceHint(priority))`

### 3. Serialization to Tiling Schedule

**Purpose**: Convert the abstract constraint solution into concrete memory operations

**Process**:
1. Solver finds valid tile dimensions that satisfy all constraints
2. `serializeTilingSolution()` computes:
   - Which memory regions to load (input cubes)
   - Where to store results (output cubes)
   - How to replace template variables for each tile

**Example Structure**:
```python
@classmethod
def serializeTilingSolution(cls, tilingSolution, absoluteOutputCubes,
                           targetMemLevel, ctxt, operatorRepresentation):
    # 1. Extract base memory addresses
    inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(...)

    # 2. For each output tile, compute corresponding input tiles
    for outputCube in outputCubes:
        inputCube = computeInputFromOutput(outputCube, ...)
        inputCubes.append(inputCube)

    # 3. Create replacement values for template variables
    replacements = {
        "tile_height": [cube.dims[1] for cube in inputCubes],
        "tile_width": [cube.dims[2] for cube in inputCubes],
        ...
    }

    # 4. Build schedules
    tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets,
                                   inputLoadSchedule, outputLoadSchedule)
    variableReplacementSchedule = VariableReplacementScheme(replacements, types)

    return variableReplacementSchedule, tilingSchedule
```

---

## How to Write Custom TileConstraints

### Step-by-Step Guide

#### Step 1: Create the Constraint Class

```python
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from typing import Dict

class MyOperatorTileConstraint(TileConstraint):
    pass  # Start with inheriting from base
```

#### Step 2: Implement Geometrical Constraints

Ask yourself:
- How do input dimensions relate to output dimensions?
- What are the mathematical formulas?

```python
@staticmethod
def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict,
                             ctxt: NetworkContext) -> TilerModel:
    # 1. Get buffer names from parseDict
    inputBufferName = parseDict['data_in']
    outputBufferName = parseDict['data_out']

    # 2. Register tensors with the model
    for bufferName in [inputBufferName, outputBufferName]:
        tilerModel.addTensorDimToModel(ctxt, bufferName)

    # 3. Get dimension variables
    inputDim = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=0)
    outputDim = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=0)

    # 4. Add constraints
    tilerModel.addConstraint(outputDim == inputDim)  # Example: 1:1 mapping

    return tilerModel
```

#### Step 3: Implement Policy Constraints

Ask yourself:
- What dimensions MUST NOT be tiled? (e.g., channels in im2col conv)
- What are minimum tile sizes? (e.g., kernel size for conv)
- What alignment is required? (e.g., SIMD width, DMA alignment)

```python
@staticmethod
def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict,
                       ctxt: NetworkContext) -> TilerModel:
    # Example policies:

    # Don't tile channels
    channelVar = tilerModel.getTensorDimVar(tensorName=inputName, dimIdx=3)
    tilerModel.addConstraint(channelVar == parseDict['num_channels'])

    # Minimum tile size
    heightVar = tilerModel.getTensorDimVar(tensorName=inputName, dimIdx=1)
    tilerModel.addConstraint(heightVar >= 8)

    # Alignment constraint
    widthVar = tilerModel.getTensorDimVar(tensorName=inputName, dimIdx=2)
    tilerModel.addConstraint((widthVar % 4) == 0)

    return tilerModel
```

#### Step 4: Implement Serialization

This is the most complex part. You need to:

1. Compute input tiles from output tiles
2. Handle edge cases (boundaries, padding)
3. Create variable replacements for code generation

```python
@classmethod
def serializeTilingSolution(cls, tilingSolution, absoluteOutputCubes,
                           targetMemLevel, ctxt, operatorRepresentation):
    outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

    # Extract addresses
    addrNames = ['data_in', 'data_out']
    inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
        tilingSolution, targetMemLevel, operatorRepresentation, addrNames)

    # Compute input tiles
    inputCubes = []
    for outputCube in outputCubes:
        # Your logic to compute input from output
        inputCube = HyperRectangle(offset=outputCube.offset,
                                   dims=outputCube.dims)
        inputCubes.append(inputCube)

    # Build schedules
    inputLoadSchedule = [{"data_in": cube} for cube in inputCubes]
    outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]

    tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets,
                                   inputLoadSchedule, outputLoadSchedule)

    # Variable replacements (if needed)
    replacements = {}
    replacementTypes = {}
    variableReplacementSchedule = VariableReplacementScheme(
        replacements, replacementTypes)

    return variableReplacementSchedule, tilingSchedule
```

#### Step 5: Register the Constraint

In your target's `__init__.py`:
```python
from .MyOperatorTileConstraint import MyOperatorTileConstraint

tileConstraints = {
    'MyOperator': MyOperatorTileConstraint,
    # ... other constraints
}
```

---

## Debugging Tiling Constraints

### Common Issues and Solutions

#### Issue 1: "No solution found"

**Symptom**: `AssertionError: Error in Tiler: No solution found`

**Causes**:
1. **Over-constrained**: Constraints are mathematically incompatible
2. **Memory too small**: No valid tiling fits in available memory

**Debug Steps**:
1. Check `debugConstraints()` output in TilerModel.py:252
2. Look for "offending constraints" in error message
3. Simplify constraints one by one
4. Increase memory size to test if memory-bound

**Example Fix**:
```python
# Instead of strict equality for performance:
tilerModel.addConstraint(var == maxValue, strategy=PerformanceHint(1))
# This constraint will be dropped if it makes the problem infeasible
```

#### Issue 2: "Minimal memory requirement violated"

**Symptom**: `ERROR: minimal memory requirement violated, please increase L1 to at least 16384`

**Causes**:
- Memory constraints are too tight
- Minimum tile sizes require more memory than available

**Solutions**:
1. Increase memory size in platform definition
2. Relax policy constraints (if safe)
3. Reduce minimum tile size requirements

#### Issue 3: Incorrect Output

**Symptom**: Results don't match expected values

**Causes**:
1. **Wrong input cube calculation**: Serialization computes wrong memory regions
2. **Padding errors**: Edge tiles not handled correctly
3. **Stride errors**: Tile offsets don't account for strides

**Debug Steps**:
1. Add logging to `serializeTilingSolution()`:
```python
for i, (inputCube, outputCube) in enumerate(zip(inputCubes, outputCubes)):
    print(f"Tile {i}:")
    print(f"  Input: offset={inputCube.offset}, dims={inputCube.dims}")
    print(f"  Output: offset={outputCube.offset}, dims={outputCube.dims}")
```

2. Verify input/output relationships manually
3. Check edge cases (first tile, last tile, remainder tiles)

### TilerModel Debugging API

The `TilerModel` class provides a `debugConstraints()` method:

```python
def debugConstraints(self) -> bool:
    offendingGeometricalConstraints: List[IntExpr] = []
    offendingMemoryConstraints: List[Tuple[MemoryLevel, IntVar, IntExpr]] = []

    # Tests each constraint individually
    for constraint in self._constraints:
        if not self._model.CheckConstraint(constraint):
            offendingGeometricalConstraints.append(constraint)

    # Reports minimal set of conflicting constraints
    if offendingGeometricalConstraints:
        raise RuntimeError(f"Infeasible constraints: {offendingGeometricalConstraints}")
```

**Usage**: Called automatically when `trySolveModel()` fails

---

## Common Patterns and Examples

### Pattern 1: Element-wise Operations (Add, Mul, etc.)

**Characteristics**:
- Input and output shapes are identical
- Can tile any dimension
- Very simple constraints

**Example**: `BOPTileConstraint.py` (Binary Operator)

```python
class BOPTileConstraint(TileConstraint):
    @classmethod
    def addGeometricalConstraint(cls, tilerModel, parseDict, ctxt):
        # All dimensions must match
        for dim in range(len(inputShape)):
            inputDim1 = tilerModel.getTensorDimVar(input1Name, dim)
            inputDim2 = tilerModel.getTensorDimVar(input2Name, dim)
            outputDim = tilerModel.getTensorDimVar(outputName, dim)

            tilerModel.addConstraint(inputDim1 == inputDim2)
            tilerModel.addConstraint(inputDim1 == outputDim)
```

### Pattern 2: Convolution Operations

**Characteristics**:
- Complex input/output relationship
- Spatial overlap between tiles
- Cannot tile channels (for im2col)
- Must respect kernel size, stride, padding

**Key Insight**: For spatial tiling, input tiles need **overlap** equal to (kernel_size - 1)

**Example**: Computing input cube from output cube (Conv2DTileConstraint.py:393)

```python
def computeInputCube(kernelShape, pads, strides, inputCSize,
                    outputCube, outputDims, inputDims):
    # Extract output tile info
    outputHOffset, outputWOffset = outputCube.offset[1:3]
    outputHSize, outputWSize = outputCube.dims[1:3]

    # Calculate per-tile padding (only at edges)
    tilePadTop = padTop if (outputHOffset == 0) else 0
    tilePadLeft = padLeft if (outputWOffset == 0) else 0
    tilePadBottom = padBottom if (outputHOffset + outputHSize == outputDims[1]) else 0
    tilePadRight = padRight if (outputWOffset + outputWSize == outputDims[2]) else 0

    # Calculate input offset (accounting for padding and stride)
    inputHOffset = max(outputHOffset * strideH - padTop, 0)
    inputWOffset = max(outputWOffset * strideW - padLeft, 0)

    # Calculate input size (worst-case: kernel overlap on both sides)
    inputHSize = outputHSize * strideH + (kernelH - 1) - (tilePadTop + tilePadBottom)
    inputWSize = outputWSize * strideW + (kernelW - 1) - (tilePadLeft + tilePadRight)

    # Clamp to actual input bounds
    inputHSize = min(inputHSize, inputDims[1] - inputHOffset)
    inputWSize = min(inputWSize, inputDims[2] - inputWOffset)

    return HyperRectangle(offset=(batch, inputHOffset, inputWOffset, 0),
                         dims=(batchSize, inputHSize, inputWSize, inputCSize))
```

### Pattern 3: Matrix Multiplication (GEMM)

**Characteristics**:
- Two inputs with different shapes
- Output shape determined by matrix dimensions
- Can tile along M, N dimensions (usually not K)

**Key Insight**: K dimension often not tiled (kept whole) to avoid partial accumulation

**Example** (simplified):
```python
# A: [M, K], B: [K, N], C: [M, N]
# Geometrical constraints:
tilerModel.addConstraint(outputM == inputAM)
tilerModel.addConstraint(outputN == inputBN)
tilerModel.addConstraint(inputAK == inputBK)

# Policy: Keep K whole (avoid partial products)
tilerModel.addConstraint(inputAK == parseDict['K'])
```

### Pattern 4: Reduction Operations (Mean, Sum)

**Characteristics**:
- Reduce along one or more dimensions
- Output dimension is 1 along reduced axes

**Example**:
```python
# Reduce along axis=1
tilerModel.addConstraint(outputDim1 == 1)
tilerModel.addConstraint(inputDim1 == parseDict['full_size'])  # Keep whole
```

---

## Target-Specific TileConstraints

### Generic Target

**Location**: `Deeploy/Targets/Generic/TileConstraints/`

**Purpose**: Reference implementations, CPU-oriented

**Examples**:
- `UnaryTileConstraint.py`: Template for single-input ops
- `BOPTileConstraint.py`: Template for binary ops
- `AddTileConstraint.py`: Simple inheritance example

**No special constraints**: Generic target has no alignment or special requirements

### PULPOpen Target

**Location**: `Deeploy/Targets/PULPOpen/TileConstraints/`

**Platform**: PULP processors (8-core RISC-V cluster)

**Special Constraints**:
- **SIMD alignment**: Some ops require 4-byte alignment
- **DMA transfers**: Prefer larger, aligned transfers
- **Im2Col Conv**: Input channels must be whole

**Key Files**:
- `ConvTileConstraint.py`: Sophisticated Conv2D tiling with spatial support
- `GEMMTileConstraint.py`: Optimized matrix multiplication
- `MaxPoolTileConstraint.py`: Pooling operations

### Neureka Target

**Location**: `Deeploy/Targets/Neureka/TileConstraints/`

**Platform**: Neural network accelerator with weight memory

**Special Constraints**:
- **Weight memory**: Weights can be in separate memory
- **Subtile size**: Neureka processes in 3x3 subtiles
- **Channel alignment**: Must be multiples of accelerator width

**Key Insight**: Weight loading strategy differs based on memory level

**Example** (NeurekaDenseConstraint.py:216):
```python
# If weights in accelerator SRAM, use offset
if weightBuffer._memoryLevel == "WeightMemory_SRAM":
    replacements['weight_addr_offset'] = [...]
else:
    # Otherwise, load weights per tile
    for cube in outputCubes:
        load['weight'] = HyperRectangle(...)
```

### Snitch Target

**Location**: `Deeploy/Targets/Snitch/TileConstraints/`

**Platform**: Snitch cluster (FP-heavy RISC-V)

**Special Constraints**:
- **FP32/FP64 support**: Different alignment for floating-point
- **DMA engines**: Optimized for specific transfer patterns

---

## Advanced Topics

### Performance Hints

Use `PerformanceHint` for **optional constraints** that improve performance but aren't required for correctness:

```python
# Prefer full-size tiles, but allow smaller if needed
tilerModel.addConstraint(inputHeightVar == inputHeightVar.Max(),
                        strategy=PerformanceHint(priority=1))
```

**Priority**: Higher priority hints are tried first. If they make the problem infeasible, they're dropped.

### Memory Hierarchy

Tiling works across **memory levels**:
- **L3** (DRAM): Slow, large
- **L2** (Shared L2 Cache): Medium speed, medium size
- **L1** (Scratchpad): Fast, small

**Constraint flow**:
1. Add constraints for each memory level
2. Solver ensures tiles fit at each level
3. Schedule generated includes transfers between levels

### Multi-buffering

For performance, use **double/triple buffering**:
- While computing on tile N, load tile N+1
- Requires more memory but hides latency

**Implementation**:
```python
memoryConstraint = MemoryConstraint(memoryLevel, size)
memoryConstraint.multiBufferCoefficient = 2  # Double buffering
```

### Variable Replacement

The `VariableReplacementScheme` allows **per-tile parameterization**:

```python
replacements = {
    "tile_height": [32, 32, 16],  # First two tiles: 32, last: 16
    "padding_top": [1, 0, 0],     # Only first tile has padding
}
```

These values replace template variables in code generation.

---

## Visualization and Tools

### Logging Tile Solutions

Add to your TileConstraint:

```python
import logging
log = logging.getLogger(__name__)

@classmethod
def serializeTilingSolution(cls, ...):
    log.debug(f"Tiling solution for {operatorRepresentation['nodeName']}:")
    for i, (inCube, outCube) in enumerate(zip(inputCubes, outputCubes)):
        log.debug(f"  Tile {i}: in={inCube}, out={outCube}")
```

Enable with: `export DEEPLOY_LOG_LEVEL=DEBUG`

### Understanding HyperRectangles

A `HyperRectangle` is an N-dimensional box defined by:
- **offset**: Starting position in each dimension `(batch, H, W, C)`
- **dims**: Size in each dimension `(batchSize, height, width, channels)`

**Example**:
```python
# Full tensor: [1, 32, 32, 64]
# Tile 0: HyperRectangle(offset=(0, 0, 0, 0), dims=(1, 16, 16, 64))
#         → Top-left quarter, all channels
# Tile 1: HyperRectangle(offset=(0, 0, 16, 0), dims=(1, 16, 16, 64))
#         → Top-right quarter, all channels
```

### Memory Layout Visualization

Use the `TilingSchedule.__repr__()` to visualize:

```python
print(tilingSchedule)
# Outputs:
# inputBaseOffsets: {'data_in': [0x1000]}
# outputBaseOffsets: {'data_out': [0x2000]}
# inputLoadSchedule:
# {'data_in': HyperRectangle(offset=(0,0,0,0), dims=(1,16,16,64))}
# {'data_in': HyperRectangle(offset=(0,0,16,0), dims=(1,16,16,64))}
# ...
```

---

## Best Practices

### 1. Start Simple

Begin with:
- Inherit from existing similar constraint
- Add minimal constraints
- Test with small networks first

### 2. Think About Edge Cases

Always consider:
- **First tile**: May need padding
- **Last tile**: May be smaller (remainder)
- **Single tile**: Full tensor fits in memory
- **Odd dimensions**: Not divisible by preferred tile size

### 3. Document Your Assumptions

Add docstrings explaining:
- Why certain dimensions can't be tiled
- What hardware constraints you're encoding
- Edge case handling

Example:
```python
@staticmethod
def addPolicyConstraint(tilerModel, parseDict, ctxt):
    """
    Policy constraints for PULP Conv2D.

    Constraints:
    1. Input channels must be whole (im2col requires full channel depth)
    2. Input spatial dims >= kernel size (need at least one kernel application)
    3. Input spatial dims divisible by stride (avoid partial stride at edges)
    4. Weight dims kept whole (loaded once, reused across tiles)

    Platform: PULP uses im2col algorithm, which requires these constraints.
    """
```

### 4. Reuse Common Patterns

Don't reinvent the wheel:
- Inherit from `BOPTileConstraint` for element-wise ops
- Reuse `Conv2DTileConstraint.computeInputCube()` for convolution-like ops
- Copy-paste-modify from similar operators

### 5. Test Thoroughly

Test your constraint with:
- **Varying memory sizes**: Force different tiling strategies
- **Different tensor shapes**: Square, rectangular, odd sizes
- **Edge cases**: Single tile, many tiles, remainder tiles

---

## Glossary

| Term | Definition |
|------|------------|
| **Tile** | A hyperrectangular subset of a tensor that fits in fast memory |
| **HyperRectangle** | N-dimensional rectangular region defined by offset and dimensions |
| **TilerModel** | OR-Tools constraint satisfaction model for finding valid tilings |
| **Geometrical Constraint** | Mathematical relationship between input/output tensor dimensions |
| **Policy Constraint** | Platform-specific requirement (alignment, minimum size, etc.) |
| **Serialization** | Converting abstract tiling solution to concrete memory operations |
| **Variable Replacement** | Per-tile substitution of values in code templates |
| **TilingSchedule** | Sequence of memory load/store operations for all tiles |
| **Memory Level** | Layer in memory hierarchy (L1/L2/L3/DRAM) |
| **Im2Col** | Image-to-column transformation for implementing convolution as GEMM |

---

## References

### Key Files

- **Base classes**: `Deeploy/TilingExtension/TileConstraint.py`
- **Solver**: `Deeploy/TilingExtension/TilerModel.py`
- **Code generation**: `Deeploy/TilingExtension/TilingCodegen.py`
- **Memory**: `Deeploy/TilingExtension/MemoryConstraints.py`

### Example Implementations

- **Simple**: `Deeploy/Targets/Generic/TileConstraints/AddTileConstraint.py`
- **Medium**: `Deeploy/Targets/Generic/TileConstraints/BOPTileConstraint.py`
- **Complex**: `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py`
- **Advanced**: `Deeploy/Targets/Neureka/TileConstraints/NeurekaDenseConstraint.py`

### External Documentation

- [OR-Tools CP-SAT Documentation](https://developers.google.com/optimization/cp/cp_solver)
- [Deeploy Main Docs](https://pulp-platform.github.io/Deeploy/)
- [PULP Platform](https://pulp-platform.org/)

---

## Changelog

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-16 | Claude | Initial comprehensive documentation of TileConstraints system |

---

*This document serves as both a reference and a tutorial for understanding and working with Deeploy's tiling constraint system. For questions or improvements, please contribute to the repository.*
