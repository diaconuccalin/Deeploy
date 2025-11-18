# LayerNorm Tiling Constraints - PULPOpen

> **File:** `Deeploy/Targets/PULPOpen/TileConstraints/LayernormTileConstraint.py`
> **Class:** `LayernormTileConstraint`
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

This file defines tiling constraints for Layer Normalization operations on the PULPOpen platform. Layer normalization normalizes across the feature dimension (last dimension) of the input tensor.

LayerNorm formula:
```
y = (x - mean(x)) / sqrt(var(x) + epsilon) * scale + bias
```

Where the mean and variance are computed over the last dimension for each sample.

The key constraint is that the **last dimension cannot be tiled** because normalization statistics (mean, variance) must be computed over the complete feature dimension.

### Supported Data Types

| Input Type | Scale Type | Bias Type | Output Type |
|------------|------------|-----------|-------------|
| `float32_t`| `float32_t`| `float32_t`| `float32_t` |

### Parent Class

- **Inherits from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_in` | `[..., D]` | Input tensor (arbitrary leading dimensions, last dim is feature) |
| `weight` | `[D]` | Scale parameter (gamma) |
| `bias` | `[D]` | Bias parameter (beta) |

### Output Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_out` | `[..., D]` | Output tensor (same shape as input) |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `weight` | `str` | Name of scale buffer in context |
| `bias` | `str` | Name of bias buffer in context |

---

## Geometrical Constraints

### Dimension Variables

The constraint works with arbitrary tensor shapes, focusing on the last dimension.

| Variable | Tensor | Dimension | Description |
|----------|--------|-----------|-------------|
| `lastDimIdx` | `data_in` | `len(shape) - 1` | Index of last dimension |
| `lastDimLen` | `data_in` | `shape[-1]` | Size of last dimension |

### Constraints

#### Last Dimension Fixed Constraint

**Proposition:**
```
inputDimVar[lastDimIdx] == lastDimLen
```

**Description:** The last dimension of the input tensor must remain at its full size (cannot be tiled). This is because normalization statistics must be computed over the entire feature dimension.

---

#### Scale Dimension Constraint

**Proposition:**
```
inputDimVar[lastDimIdx] == scaleDimVar[0]
```

**Description:** The scale (weight) tensor dimension must match the last dimension of the input. Scale has shape `[D]` where D is the feature dimension.

---

#### Bias Dimension Constraint

**Proposition:**
```
inputDimVar[lastDimIdx] == biasDimVar[0]
```

**Description:** The bias tensor dimension must match the last dimension of the input.

---

#### Input-Output Shape Match

**Propositions:**
```python
for idx, dim in enumerate(inputShape):
    inputDimVar[idx] == outputDimVar[idx]
```

**Description:** Input and output tensors must have identical shapes. LayerNorm preserves tensor shape.

---

## Policy Constraints

**Note:** The `LayernormTileConstraint` class does not define any explicit policy constraints beyond the geometrical constraints. The key policy decision (no tiling on last dimension) is enforced through the geometrical constraint.

### Implicit Policy

The geometrical constraint `inputDimVar[lastDimIdx] == lastDimLen` effectively acts as a policy constraint, ensuring:
- Complete feature dimension for accurate normalization
- Full scale and bias vectors can be applied
- No partial reduction results

---

## Tiling Schedule

### Output Cube Processing

For each output cube, the tiling schedule creates corresponding input and parameter cubes:

```python
for cube in outputCubes:
    # Compute total size for this tile
    newSize = np.prod(cube.dims)

    # Scale and bias always use full feature dimension
    # with offset 0 (parameters are shared across tiles)
    weightCube = HyperRectangle((0,), (cube.dims[-1],))
    biasCube = HyperRectangle((0,), (cube.dims[-1],))

    # Input cube matches output cube exactly
    inputLoadSchedule.append({
        "data_in": cube,
        "weight": weightCube,
        "bias": biasCube
    })
    outputLoadSchedule.append({"data_out": cube})
```

### Parameter Handling

Scale and bias tensors always:
- Start at offset 0 (no tiling on parameters)
- Have size equal to `cube.dims[-1]` (full feature dimension)
- Are loaded completely for each tile

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": InputCube, "weight": ScaleCube, "bias": BiasCube},
    ...
]

outputLoadSchedule = [
    {"data_out": OutputCube},
    ...
]
```

---

## Replacements

Variables that are replaced with tile-specific values during code generation.

| Variable | Type | Description |
|----------|------|-------------|
| `size` | `uint16_t` | Total number of elements in the tile (`np.prod(cube.dims)`) |

**Note:** The `size` replacement represents the total elements to process, which is the product of all tile dimensions.

---

## Visual Explanation

### LayerNorm Computation

```
Input: [Batch, Seq, Features]
          |     |      |
          v     v      v
       +------------------+
       | For each sample: |
       |   mean, var      |
       |   over Features  |
       +------------------+
              |
              v
       Normalize + Scale + Bias
              |
              v
Output: [Batch, Seq, Features]
```

### Tiling Strategy

```
Input Tensor [B, S, D] where D = feature dimension

Allowed Tiling:
+---------------------------+
|  Tile 0: [B_tile, S_tile, D]  | <- Full D
+---------------------------+
|  Tile 1: [B_tile, S_tile, D]  | <- Full D
+---------------------------+

NOT Allowed:
+-------------+-------------+
|   [B, S, D/2]  |  [B, S, D/2]  |  <- Split D
+-------------+-------------+
   INVALID: Cannot compute proper mean/var
```

### Transformer Context

```
LayerNorm in Transformer:

Input:  [Batch, SeqLen, EmbedDim]
              |
        +-----+-----+
        |           |
    (Attention)  (FFN)
        |           |
        +-----+-----+
              |
         LayerNorm
              |
Output: [Batch, SeqLen, EmbedDim]

Tiling: Can split Batch and SeqLen, NOT EmbedDim
```

---

## Examples

### Example 1: Simple LayerNorm Tiling

**Input Parameters:**
- Input shape: `[4, 128, 512]` (Batch=4, Seq=128, Embed=512)
- Scale shape: `[512]`
- Bias shape: `[512]`
- L1 Memory: 32KB

**Tiling Result:**

Since feature dimension (512) cannot be tiled:
- Output tile 0: Shape `[2, 64, 512]`, Offset `[0, 0, 0]`
- Output tile 1: Shape `[2, 64, 512]`, Offset `[0, 64, 0]`
- Output tile 2: Shape `[2, 64, 512]`, Offset `[2, 0, 0]`
- Output tile 3: Shape `[2, 64, 512]`, Offset `[2, 64, 0]`

**Memory per tile:**
- Input: 2 * 64 * 512 * 4 bytes = 256 KB (may need more tiling)
- Scale: 512 * 4 bytes = 2 KB
- Bias: 512 * 4 bytes = 2 KB

---

### Example 2: 2D LayerNorm

**Input Parameters:**
- Input shape: `[32, 1024]` (Batch=32, Features=1024)
- Scale shape: `[1024]`
- Bias shape: `[1024]`

**Tiling Result:**

Only batch dimension can be tiled:
- Tile 0: Shape `[8, 1024]`, Offset `[0, 0]`
- Tile 1: Shape `[8, 1024]`, Offset `[8, 0]`
- Tile 2: Shape `[8, 1024]`, Offset `[16, 0]`
- Tile 3: Shape `[8, 1024]`, Offset `[24, 0]`

---

### Example 3: Large Feature Dimension

**Scenario:** Feature dimension too large for L1

**Input Parameters:**
- Input shape: `[1, 128, 4096]` (Large embedding)
- L1 Memory: 32KB

**Problem:**
- Single sample row: 4096 * 4 = 16 KB
- Scale: 4096 * 4 = 16 KB
- Bias: 4096 * 4 = 16 KB
- Total minimum: ~48 KB > 32 KB

**Solution:** This case requires:
1. Larger L1 allocation
2. Multi-level tiling (L1/L2/L3)
3. Or restructured network (smaller embedding)

---

## Notes and Limitations

### Known Limitations

1. **No feature dimension tiling:** The last dimension (feature/embedding dimension) cannot be tiled because:
   - Mean and variance must be computed over complete features
   - Partial statistics would give incorrect results

2. **Memory pressure from large features:** Large embedding dimensions (common in transformers) require significant L1 memory:
   - Input tile + Scale + Bias must all fit
   - Minimum: feature_dim * element_size * 3

3. **Parameter reload:** Scale and bias are reloaded for each tile since they're always full-sized.

### Platform-Specific Notes

- **Memory alignment:** Scale and bias tensors should be 4-byte aligned
- **DMA efficiency:** Since parameters are small, consider caching them in L1

### Performance Considerations

1. **Maximize leading dimension tiles:** Since feature dimension is fixed, maximize tiling on batch and sequence dimensions to minimize parameter reload overhead.

2. **Parameter caching:** Scale and bias are the same for all tiles. If memory allows, keep them resident in L1.

3. **Memory bandwidth:**
   - LayerNorm has moderate arithmetic intensity
   - Three passes over data: mean, variance, normalize
   - Optimize for memory transfer of input/output tiles

4. **Vectorization:** Feature dimension is typically large and can benefit from SIMD operations for mean/variance/normalize.

### Memory Estimation

For a tile of shape `[..., D]`:
```
Required L1 Memory >=
    input_tile_size * sizeof(input_type) +
    D * sizeof(scale_type) +
    D * sizeof(bias_type) +
    output_tile_size * sizeof(output_type)
```

Since input and output shapes match:
```
Required >= 2 * tile_elements * element_size + 2 * D * element_size
```

---

## References

- Source file: `Deeploy/Targets/PULPOpen/TileConstraints/LayernormTileConstraint.py`
- Related templates: `Deeploy/Targets/PULPOpen/Templates/`
- Tiler configuration: `Deeploy/Targets/PULPOpen/Tiler.py`
- LayerNorm paper: "Layer Normalization" (Ba et al., 2016)
