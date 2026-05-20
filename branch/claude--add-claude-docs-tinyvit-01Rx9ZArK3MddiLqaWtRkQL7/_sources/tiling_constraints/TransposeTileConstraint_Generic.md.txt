# Transpose Tiling Constraints - Generic

> **File:** `Deeploy/Targets/Generic/TileConstraints/TransposeTileConstraint.py`
> **Class:** `TransposeTileConstraint`
> **Last Updated:** 2025-11-18

---

## Table of Contents

1. [Overview](#overview)
2. [Tensor Definitions](#tensor-definitions)
3. [Geometrical Constraints](#geometrical-constraints)
4. [Policy Constraints](#policy-constraints)
5. [Tiling Schedule](#tiling-schedule)
6. [Replacements](#replacements)
7. [Visual Explanation](#visual-explanation)
8. [Examples](#examples)
9. [Notes and Limitations](#notes-and-limitations)

---

## Overview

### Description

The `TransposeTileConstraint` handles tensor transposition (dimension permutation) operations. It establishes the relationship between input and output dimensions based on a permutation vector and computes input cubes by applying the inverse permutation to output cubes.

### Supported Operations

- ONNX Transpose operator
- Dimension permutation/reordering

### Parent Class

- **Inherits from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_in` | `[D_0, D_1, ..., D_n]` | Input tensor before transposition |

### Output Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_out` | `[D_perm[0], D_perm[1], ..., D_perm[n]]` | Transposed output tensor |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `perm` | `List[int]` | Permutation vector (e.g., `[0, 2, 1]` swaps dims 1 and 2) |

---

## Geometrical Constraints

Geometrical constraints define how input and output dimensions relate through the permutation.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `inputDimVar[i]` | `data_in` | i | Dimension i of input tensor |
| `outputDimVar[i]` | `data_out` | i | Dimension i of output tensor |

### Constraints

#### Permutation Constraint

**Proposition:**
```
For all i in [0, N-1]:
    outputDimVar[i] == inputDimVar[perm[i]]
```

**Description:** Each output dimension corresponds to an input dimension as specified by the permutation vector. If `perm = [0, 2, 1]`, then:
- `outputDim[0] = inputDim[0]`
- `outputDim[1] = inputDim[2]`
- `outputDim[2] = inputDim[1]`

---

## Policy Constraints

**None.** The `TransposeTileConstraint` does not impose additional policy constraints. Tiling can occur along any dimension.

---

## Tiling Schedule

### Input Cube Computation

The input cube for each output tile is computed by applying the inverse permutation to the output cube dimensions and offsets.

**Algorithm:**
```python
invPerm = invertPermutation(perm)

for output_cube in output_cubes:
    input_cube = permuteHyperRectangle(output_cube, invPerm)
    # input_cube.dims[i] = output_cube.dims[invPerm[i]]
    # input_cube.offset[i] = output_cube.offset[invPerm[i]]
```

**Example:**
- Permutation: `[0, 2, 1]` (swap dims 1 and 2)
- Inverse permutation: `[0, 2, 1]` (same in this case)
- Output cube: dims `[1, 8, 16]`, offset `[0, 0, 0]`
- Input cube: dims `[1, 16, 8]`, offset `[0, 0, 0]`

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": InCube},  # Permuted from output cube
    ...
]

outputLoadSchedule = [
    {"data_out": OutCube},
    ...
]
```

---

## Replacements

Variables that are replaced with tile-specific values during code generation.

| Variable | Type | Description |
|----------|------|-------------|
| `dimLen_0` | `uint16_t*` | Input dimension 0 for each tile |
| `dimLen_1` | `uint16_t*` | Input dimension 1 for each tile |
| ... | ... | ... |
| `dimLen_{N-1}` | `uint16_t*` | Input dimension N-1 for each tile |

### Replacement Computation

```python
numDims = len(input_shape)

for i in range(numDims):
    replacements[f"dimLen_{i}"] = []

for output_cube in output_cubes:
    input_cube = permuteHyperRectangle(output_cube, invPerm)
    for i, dim in enumerate(input_cube.dims):
        replacements[f"dimLen_{i}"].append(dim)
```

---

## Visual Explanation

### Transpose Operation

```
Input [2, 3, 4]                    Output [2, 4, 3]
perm = [0, 2, 1]

  Dim 0   Dim 1   Dim 2            Dim 0   Dim 1   Dim 2
    2   x   3   x   4       ->       2   x   4   x   3

    Input dims [2,3,4]              Output dims [2,4,3]
    invPerm = [0, 2, 1]
```

### Tile Mapping

```
Output Tile                        Input Tile (via inverse perm)
┌─────────────┐                    ┌─────────────┐
│ dims=[1,4,3]│      invPerm       │ dims=[1,3,4]│
│ off=[0,0,0] │  ─────────────>    │ off=[0,0,0] │
└─────────────┘                    └─────────────┘

Output element at (n, h, w)  <->  Input element at (n, w, h)
```

---

## Examples

### Example 1: Simple Dimension Swap (3D)

**Input Parameters:**
- Input shape: `[1, 64, 128]`
- Permutation: `[0, 2, 1]` (swap dims 1 and 2)
- Output shape: `[1, 128, 64]`

**Tiling (along output dim 1):**
- Output Tile 0: Shape `[1, 64, 64]`, Offset `[0, 0, 0]`
- Output Tile 1: Shape `[1, 64, 64]`, Offset `[0, 64, 0]`

**Input Tiles (via inverse permutation):**
- Input Tile 0: Shape `[1, 64, 64]`, Offset `[0, 0, 0]`
- Input Tile 1: Shape `[1, 64, 64]`, Offset `[0, 0, 64]`

**Replacements for Tile 0:**
- `dimLen_0 = 1`
- `dimLen_1 = 64`
- `dimLen_2 = 64`

---

### Example 2: NHWC to NCHW Conversion (4D)

**Input Parameters:**
- Input shape: `[1, 32, 32, 64]` (NHWC)
- Permutation: `[0, 3, 1, 2]` (NHWC -> NCHW)
- Output shape: `[1, 64, 32, 32]` (NCHW)

**Inverse Permutation:** `[0, 2, 3, 1]`

**Tiling (along output channel dimension):**
- Output Tile 0: Shape `[1, 32, 32, 32]`, Offset `[0, 0, 0, 0]`
- Output Tile 1: Shape `[1, 32, 32, 32]`, Offset `[0, 32, 0, 0]`

**Input Tiles:**
- Input Tile 0: Shape `[1, 32, 32, 32]`, Offset `[0, 0, 0, 0]`
- Input Tile 1: Shape `[1, 32, 32, 32]`, Offset `[0, 0, 0, 32]`

Note: Input tile offsets are permuted from output tile offsets.

**Replacements for Tile 0:**
- `dimLen_0 = 1`
- `dimLen_1 = 32`
- `dimLen_2 = 32`
- `dimLen_3 = 32`

---

### Example 3: No Tiling Required

**Input Parameters:**
- Input shape: `[2, 4, 8]`
- Permutation: `[2, 1, 0]` (reverse all dimensions)
- Output shape: `[8, 4, 2]`
- L1 Memory: Sufficient for full tensor

**Result:**
- Single tile covering entire tensor
- Input cube: `[2, 4, 8]`
- Output cube: `[8, 4, 2]`

**Replacements:**
- `dimLen_0 = 2`
- `dimLen_1 = 4`
- `dimLen_2 = 8`

---

## Notes and Limitations

### Key Characteristics

1. **Flexible tiling:** Can tile along any dimension since the permutation relationship is maintained per-tile.

2. **Non-contiguous memory access:** Transpose operations typically involve strided memory access patterns, which can be less efficient than contiguous access.

3. **Dimension count preserved:** Input and output always have the same number of dimensions.

### Performance Considerations

1. **Memory access patterns:** Transposing can cause cache-unfriendly access. Larger tiles may help amortize overhead.

2. **Tile size selection:** Consider the target platform's cache line size and memory alignment.

3. **Avoid small inner dimensions:** When the innermost dimension is small after transposition, memory bandwidth utilization may be poor.

### Common Use Cases

- Layout conversion (NHWC <-> NCHW)
- Matrix transpose for GEMM optimization
- Attention mechanism reshape operations
- Preparing data for specific kernel requirements

### Platform-Specific Notes

- Some platforms may have optimized transpose kernels for specific permutations (e.g., 2D matrix transpose).
- DMA engines may support strided transfers that can accelerate certain transposition patterns.

---

## References

- Source file: `Deeploy/Targets/Generic/TileConstraints/TransposeTileConstraint.py`
- Helper functions: `_invertPermutation`, `_permuteHyperRectangle` from `LoweringOptimizationPasses.py`
- Related operators: ONNX Transpose
