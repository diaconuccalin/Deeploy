# ReduceMean Tiling Constraints - PULPOpen

> **File:** `Deeploy/Targets/PULPOpen/TileConstraints/ReduceMeanConstraint.py`
> **Class:** `ReduceMeanTileConstraint`
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

This file defines tiling constraints for ReduceMean operations on the PULPOpen platform. ReduceMean computes the mean value along specified axes of the input tensor.

The operation reduces dimensions by computing:
```
output = mean(input, axis=axes, keepdims=keepdims)
```

Key characteristics:
- Reduction can occur on any subset of axes
- `keepdims` flag controls whether reduced dimensions are kept (size 1) or removed
- Reduced axes cannot be tiled (need complete data for mean computation)

### Supported Data Types

| Input Type | Output Type |
|------------|-------------|
| `float32_t`| `float32_t` |
| `int32_t`  | `int32_t`   |

### Parent Class

- **Inherits from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_in` | `[D0, D1, ..., Dn]` | Input tensor with arbitrary dimensions |

### Output Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_out` | Depends on axes and keepdims | Reduced tensor |

**Output shape determination:**
- If `keepdims=True`: Reduced axes become size 1
- If `keepdims=False`: Reduced axes are removed

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `data_in_shape` | `List[int]` | Original input shape |
| `data_out_shape` | `List[int]` | Output shape |
| `axes` | `List[int]` | Axes along which to compute mean |
| `keepdims` | `bool` | Whether to keep reduced dimensions |

---

## Geometrical Constraints

### Constraint Logic

The geometrical constraints handle the relationship between input and output dimensions based on which axes are reduced.

#### Dimension Relationship Constraints

**Algorithm:**
```python
input_ax = 0
for idx in range(len(outputShape)):
    outputDimensionVar = getTensorDimVar(outputBuffer, idx)

    if idx in reduceAxes:
        # This axis is reduced
        if keepDims:
            # Reduced dimension becomes 1
            addConstraint(outputDimensionVar == 1)
            input_ax += 1
    else:
        # Non-reduced axis: input and output must match
        inputDimensionVar = getTensorDimVar(inputBuffer, input_ax)
        addConstraint(outputDimensionVar == inputDimensionVar)
        input_ax += 1
```

### Constraints by Case

#### Case 1: Reduced Axis with keepdims=True

**Proposition:**
```
outputDimVar[reduced_axis] == 1
```

**Description:** Reduced axes are constrained to size 1 in the output.

---

#### Case 2: Non-Reduced Axis

**Proposition:**
```
outputDimVar[output_idx] == inputDimVar[input_idx]
```

**Description:** Non-reduced axes must have matching dimensions between input and output, and can be tiled together.

---

## Policy Constraints

**Note:** The `ReduceMeanTileConstraint` class does not define explicit policy constraints. The `addPolicyConstraint` method returns the model unchanged.

### Implicit Policy Through Geometry

The key policy decisions are enforced through the input cube computation:
- Reduced axes use full input dimensions (from `originalInputShape`)
- Non-reduced axes use tiled dimensions (from output cube)

This ensures complete data is available for mean computation on reduced axes.

---

## Tiling Schedule

### Input Cube Computation

The constraint provides a dedicated method `computeInputCubeFromOutputCube` for computing input tiles:

```python
@staticmethod
def computeInputCubeFromOutputCube(outputCube, parseDict):
    originalInputShape = parseDict['data_in_shape']
    keepDims = parseDict['keepdims']

    # Start with full input dimensions
    in_cube_dims = list(originalInputShape).copy()
    in_cube_offset = [0] * len(in_cube_dims)

    out_idx = 0
    for ax in range(len(in_cube_dims)):
        if ax in parseDict['axes']:
            # Reduced axis: keep full dimension, offset 0
            # (dims already set to full size)
            if keepDims:
                out_idx += 1
        else:
            # Non-reduced axis: copy from output cube
            in_cube_dims[ax] = outputCube.dims[out_idx]
            in_cube_offset[ax] = outputCube.offset[out_idx]
            out_idx += 1

    return HyperRectangle(tuple(in_cube_offset), tuple(in_cube_dims))
```

### Key Insight

For reduced axes:
- **Offset:** Always 0 (need data from start)
- **Size:** Full original dimension (need all data for mean)

For non-reduced axes:
- **Offset:** From output cube (tiling position)
- **Size:** From output cube (tiled dimension)

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": InputCube},
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
| `data_in_shape` | `List[uint16_t]` | Input tile shape (list of 4 values) |
| `data_out_shape` | `List[uint16_t]` | Output tile shape (list of 4 values) |
| `size` | `uint16_t` | Total output elements (`np.prod(output_dims)`) |

**Note:** Shape replacements use lists of `uint16_t*` pointers, one per dimension (up to 4 dimensions supported).

---

## Visual Explanation

### ReduceMean Operation

```
Input: [4, 8, 16]

ReduceMean(axes=[1], keepdims=True):
+--------+--------+--------+
|        |        |        |
| dim 0  | dim 1  | dim 2  |  Input
| (4)    | (8)    | (16)   |
+--------+--------+--------+
              |
              v  Reduce
         +--------+
         | mean   |
         +--------+
              |
              v
+--------+--------+--------+
|        |        |        |
| dim 0  | dim 1  | dim 2  |  Output
| (4)    | (1)    | (16)   |
+--------+--------+--------+
```

### Tiling Strategy

```
Input: [B, H, W, C], reduce axes=[2] (W), keepdims=True
Output: [B, H, 1, C]

Tiling on non-reduced axes (B, H, C):
+----------------------------------+
|  Input Tile: [B_t, H_t, W, C_t]  |  <- Full W (reduced)
+----------------------------------+
              |
              v
+----------------------------------+
|  Output Tile: [B_t, H_t, 1, C_t] |
+----------------------------------+

Key: Reduced axis W uses full dimension in input
```

### keepdims Comparison

```
Input: [4, 8, 16]
axes = [1]

keepdims=True:              keepdims=False:
Output: [4, 1, 16]          Output: [4, 16]
     ^                           ^
     |                           |
  Axis kept                   Axis removed
  (size 1)                    from shape
```

---

## Examples

### Example 1: Channel-wise Mean (Global Average Pooling)

**Input Parameters:**
- Input shape: `[1, 32, 32, 128]` (NHWC)
- Axes: `[1, 2]` (reduce H and W)
- keepdims: `True`
- Output shape: `[1, 1, 1, 128]`

**Tiling Result:**

Only batch and channel can be tiled (H and W are reduced):
- Output tile 0: Shape `[1, 1, 1, 64]`, Offset `[0, 0, 0, 0]`
- Output tile 1: Shape `[1, 1, 1, 64]`, Offset `[0, 0, 0, 64]`

**Input Tiles:**
- Input tile 0: Shape `[1, 32, 32, 64]`, Offset `[0, 0, 0, 0]`
- Input tile 1: Shape `[1, 32, 32, 64]`, Offset `[0, 0, 0, 64]`

Full H=32 and W=32 required for each input tile.

---

### Example 2: Batch Mean

**Input Parameters:**
- Input shape: `[16, 256]`
- Axes: `[0]` (reduce batch)
- keepdims: `False`
- Output shape: `[256]`

**Tiling Result:**

Only feature dimension can be tiled:
- Output tile 0: Shape `[128]`, Offset `[0]`
- Output tile 1: Shape `[128]`, Offset `[128]`

**Input Tiles:**
- Input tile 0: Shape `[16, 128]`, Offset `[0, 0]`
- Input tile 1: Shape `[16, 128]`, Offset `[0, 128]`

Full batch (16) required for each input tile.

---

### Example 3: Multiple Non-Adjacent Axes

**Input Parameters:**
- Input shape: `[4, 8, 16, 32]`
- Axes: `[0, 2]` (reduce first and third)
- keepdims: `True`
- Output shape: `[1, 8, 1, 32]`

**Input Cube Computation for tile:**
```python
Output cube: offset=[0, 4, 0, 16], dims=[1, 4, 1, 16]

Input cube: offset=[0, 4, 0, 16], dims=[4, 4, 16, 16]
                    ^     ^            ^      ^
                    |     |            |      |
              Full (reduced)    Full (reduced)
```

---

## Notes and Limitations

### Known Limitations

1. **No tiling on reduced axes:** Axes being reduced must have full data available:
   - Cannot compute partial means and combine later
   - Memory requirement scales with product of reduced dimensions

2. **Shape list limitation:** Replacement types assume maximum 4 dimensions for shape arrays.

3. **Memory pressure:** When multiple large dimensions are reduced, significant memory is required:
   ```
   Input tile size = tiled_dims * prod(reduced_dims) * element_size
   ```

### Platform-Specific Notes

- **Reduction efficiency:** Mean computation requires sum followed by division - ensure accumulator precision
- **Memory layout:** Input cube preserves original axis ordering

### Performance Considerations

1. **Minimize reduced dimensions:** Reduce smaller dimensions when possible to decrease memory requirements.

2. **Tile non-reduced dimensions aggressively:** Since reduced dimensions cannot be tiled, maximize tiling on other dimensions.

3. **Memory estimation:**
   ```
   Input tile memory = prod(tiled_dims) * prod(reduced_dims) * sizeof(input)
   Output tile memory = prod(tiled_dims) * sizeof(output)
   Ratio = prod(reduced_dims) : 1
   ```

4. **Computation pattern:** Each output element requires reading `prod(reduced_dims)` input elements.

### Common Use Cases

| Use Case | Input Shape | Axes | keepdims | Output Shape |
|----------|-------------|------|----------|--------------|
| Global Avg Pool | [N,H,W,C] | [1,2] | True | [N,1,1,C] |
| Batch Mean | [N,D] | [0] | False | [D] |
| Feature Mean | [N,D] | [1] | True | [N,1] |
| Spatial Mean | [N,H,W,C] | [1,2] | False | [N,C] |

---

## References

- Source file: `Deeploy/Targets/PULPOpen/TileConstraints/ReduceMeanConstraint.py`
- Related templates: `Deeploy/Targets/PULPOpen/Templates/`
- Tiler configuration: `Deeploy/Targets/PULPOpen/Tiler.py`
- ONNX ReduceMean specification: https://onnx.ai/onnx/operators/onnx__ReduceMean.html
