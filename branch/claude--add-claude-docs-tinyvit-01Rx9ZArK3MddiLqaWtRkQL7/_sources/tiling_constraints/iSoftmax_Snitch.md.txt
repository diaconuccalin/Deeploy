# iSoftmax Tiling Constraints - Snitch

> **File:** `Deeploy/Targets/Snitch/TileConstraints/iSoftmaxTileConstraint.py`
> **Class:** `iSoftmaxTileConstraint`
> **Last Updated:** 2025-11-18

---

## Table of Contents

1. [Overview](#overview)
2. [Tensor Definitions](#tensor-definitions)
3. [Geometrical Constraints](#geometrical-constraints)
4. [Policy Constraints](#policy-constraints)
5. [Tiling Schedule](#tiling-schedule)
6. [Replacements](#replacements)
7. [Notes and Limitations](#notes-and-limitations)

---

## Overview

### Description

This constraint class defines tiling rules for the integer/incremental Softmax operation on the Snitch platform. The iSoftmax operation computes the softmax function over the last dimension of the input tensor. The constraint ensures that the last dimension (reduction axis) is not tiled, as softmax requires the complete sequence for normalization.

### Supported Data Types

| Input Type | Output Type |
|------------|-------------|
| `int8_t`   | `int8_t`    |
| `int32_t`  | `int32_t`   |
| `float32_t`| `float32_t` |

### Parent Class

- **Extends:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[d0, d1, ..., dn]` | N-dimensional | Input tensor (any dimensionality) |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[d0, d1, ..., dn]` | N-dimensional | Output tensor (same shape as input) |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions that must hold for correct operation.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `inputDim[i]` | `data_in` | i | Input dimension at index i |
| `outputDim[i]` | `data_out` | i | Output dimension at index i |

### Constraints

#### Element-wise Dimension Constraint

**Proposition:**
```
For all i in range(shapeLen):
    outputDim[i] == inputDim[i]
```

**Description:** This is an element-wise operation (softmax). All input and output dimensions must match exactly. The output tensor has the same shape as the input tensor.

---

## Policy Constraints

Policy constraints define restrictions required for the softmax algorithm to produce correct results.

### Constraints

#### Full Last Dimension Constraint

**Proposition:**
```
lastDimVar == lastDimLength
```

**Where:**
- `lastDimVar` = `tilerModel.getTensorDimVar(inputBufferName, lastDimIdx)`
- `lastDimLength` = `inputBuffer.shape[-1]`
- `lastDimIdx` = `len(inputBuffer.shape) - 1`

**Rationale:** The softmax operation computes normalized exponentials over the last dimension. This requires:
1. Finding the maximum value across the last dimension
2. Computing exponentials relative to the maximum
3. Summing all exponentials for normalization

Tiling the last dimension would prevent correct normalization since partial sums would not equal the full normalization constant.

**Priority:** Hard constraint (must be satisfied)

---

## Tiling Schedule

### Input Cube Computation

Since iSoftmax is element-wise with the constraint that the last dimension must be full, input and output cubes are identical:

**Algorithm:**

```python
for each output tile cube:
    # Input cube matches output cube exactly
    inputCube = outputCube
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": outputCube},  # Same as output cube
    ...
]

outputLoadSchedule = [
    {"data_out": outputCube},
    ...
]
```

---

## Replacements

Variables that are replaced with tile-specific values during code generation.

| Variable | Type | Description |
|----------|------|-------------|
| `lastDimLength` | `uint32_t*` | Length of the last dimension (always full) |
| `size` | `uint32_t*` | Total number of elements in the tile (product of all dimensions) |

### Symbolic Node Representation

The constraint also constructs a symbolic representation that includes:

```python
symbolicParseDict['lastDimLength'] = tilerModel.getTensorDimVar(inputBuffer.name, lastDimIdx)
```

---

## Notes and Limitations

### Known Limitations

1. **No tiling on last dimension:** The last dimension (softmax axis) cannot be tiled. The complete sequence must fit in memory for correct normalization.

2. **Memory requirements:** For large last dimensions, ensure sufficient L1 memory to hold the complete last dimension of both input and output.

3. **Flexible dimensionality:** The constraint supports tensors of any dimensionality, tiling on all dimensions except the last.

### Platform-Specific Notes

- **Snitch cluster:** The operation can be parallelized over the outer dimensions (all except the last).
- **Reduction operations:** The max-finding and sum-reduction over the last dimension are performed sequentially within each tile.

### Performance Considerations

1. **Maximize outer dimension tiles** to improve parallelization across Snitch cores
2. **Ensure last dimension fits in L1** - this is a hard constraint
3. **Consider memory bandwidth** when choosing tile sizes for outer dimensions
4. **Tile size calculation:** The `size` replacement is `np.prod(cube.dims)`, representing total elements per tile

### Tiling Strategy

```
Input Tensor [B, S, D]
+---------------------------+
|                           |
|   +---------------+       |
|   | Tile 0        |  <-- Can tile B, S dimensions
|   | D is full     |
|   +---------------+       |
|        |                  |
|   +---------------+       |
|   | Tile 1        |       |
|   | D is full     |       |
|   +---------------+       |
|                           |
+---------------------------+

Last dimension D cannot be tiled (softmax axis)
```

---

## References

- Source file: `Deeploy/Targets/Snitch/TileConstraints/iSoftmaxTileConstraint.py`
- Related templates: `Deeploy/Targets/Snitch/Templates/iSoftmaxTemplate.py`
- Tiler configuration: `Deeploy/Targets/Snitch/Tiler.py`
