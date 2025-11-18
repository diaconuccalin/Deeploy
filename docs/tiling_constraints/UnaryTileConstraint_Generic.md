# Unary Tiling Constraints - Generic

> **File:** `Deeploy/Targets/Generic/TileConstraints/UnaryTileConstraint.py`
> **Class:** `UnaryTileConstraint`
> **Last Updated:** 2025-11-18

---

## Table of Contents

1. [Overview](#overview)
2. [Tensor Definitions](#tensor-definitions)
3. [Geometrical Constraints](#geometrical-constraints)
4. [Policy Constraints](#policy-constraints)
5. [Tiling Schedule](#tiling-schedule)
6. [Replacements](#replacements)
7. [Examples](#examples)
8. [Notes and Limitations](#notes-and-limitations)

---

## Overview

### Description

The `UnaryTileConstraint` handles element-wise unary operations where the input and output tensors have identical shapes. This constraint is used for operators like activation functions (ReLU, Sigmoid, Tanh), element-wise negation, absolute value, and similar single-input, single-output operations that process each element independently.

### Supported Operations

- Activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, etc.)
- Element-wise unary math operations (Abs, Neg, Sqrt, Exp, Log, etc.)
- Type casting operations
- Clip operations

### Parent Class

- **Inherits from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_in` | `[D_0, D_1, ..., D_n]` | Input tensor of arbitrary shape |

### Output Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_out` | `[D_0, D_1, ..., D_n]` | Output tensor with same shape as input |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions that must hold for correct operation.

### Dimension Variables

For an N-dimensional tensor, the following variables are created:

| Variable | Tensor | Description |
|----------|--------|-------------|
| `inputDim1Var[i]` | `data_in` | Dimension i of input tensor |
| `outputDimVar[i]` | `data_out` | Dimension i of output tensor |

### Constraints

#### Shape Equality Constraint

**Proposition:**
```
For all i in [0, N-1]:
    inputDim1Var[i] == outputDimVar[i]
```

**Description:** All dimensions of the input tensor must equal the corresponding dimensions of the output tensor. This ensures the element-wise operation maintains shape consistency.

---

## Policy Constraints

**None.** The `UnaryTileConstraint` does not impose any policy constraints. The tiler is free to tile along any dimension.

---

## Tiling Schedule

### Input Cube Computation

For unary operations, the input cube is identical to the output cube since the operation is element-wise with no dependencies between elements.

**Algorithm:**
```python
for output_cube in output_cubes:
    input_cube = output_cube  # Direct mapping
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": Cube},
    ...
]

outputLoadSchedule = [
    {"data_out": Cube},
    ...
]
```

Where each input cube has the same dimensions and offsets as its corresponding output cube.

---

## Replacements

Variables that are replaced with tile-specific values during code generation.

| Variable | Type | Description |
|----------|------|-------------|
| `size` | `uint16_t*` | Total number of elements in the tile (`prod(cube.dims)`) |

### Replacement Computation

```python
for cube in outputCubes:
    size = np.prod(cube.dims)  # Product of all tile dimensions
```

---

## Examples

### Example 1: 2D Tensor Tiling

**Input Parameters:**
- Input shape: `[64, 128]`
- L1 Memory: 4KB (can hold ~2000 elements at 16-bit)

**Possible Tiling:**
- Tile 0: Shape `[32, 128]`, Offset `[0, 0]`, Size = 4096
- Tile 1: Shape `[32, 128]`, Offset `[32, 0]`, Size = 4096

**Replacements per tile:**
- Tile 0: `size = 4096`
- Tile 1: `size = 4096`

---

### Example 2: 4D Tensor (NHWC) Tiling

**Input Parameters:**
- Input shape: `[1, 32, 32, 64]`
- L1 Memory: 16KB

**Possible Tiling:**
- Tile 0: Shape `[1, 16, 32, 64]`, Offset `[0, 0, 0, 0]`, Size = 32768
- Tile 1: Shape `[1, 16, 32, 64]`, Offset `[0, 16, 0, 0]`, Size = 32768

**Key Property:** Input and output tiles are identical cubes.

---

### Example 3: Minimal Tiling (Full Tensor)

**Input Parameters:**
- Input shape: `[1, 8, 8, 32]`
- L1 Memory: 8KB (sufficient for full tensor)

**Result:**
- Single tile covering entire tensor
- Tile 0: Shape `[1, 8, 8, 32]`, Offset `[0, 0, 0, 0]`, Size = 2048

---

## Notes and Limitations

### Key Characteristics

1. **No tile overlap:** Since operations are element-wise, there is no redundant data transfer between tiles.

2. **Flexible tiling:** Can tile along any dimension without constraints on tile boundaries.

3. **Shape preservation:** Input and output always have identical shapes.

### Performance Considerations

1. **Prefer larger tiles** to reduce loop overhead and improve memory bandwidth utilization.

2. **Align tile boundaries** to memory alignment requirements when possible.

3. **Consider vectorization** - tile sizes that are multiples of SIMD width improve performance.

### Common Use Cases

- Post-convolution activation functions
- Normalization operations
- Element-wise transformations in attention mechanisms

---

## References

- Source file: `Deeploy/Targets/Generic/TileConstraints/UnaryTileConstraint.py`
- Related operators: ReLU, Sigmoid, Tanh, Abs, Neg, Clip, etc.
