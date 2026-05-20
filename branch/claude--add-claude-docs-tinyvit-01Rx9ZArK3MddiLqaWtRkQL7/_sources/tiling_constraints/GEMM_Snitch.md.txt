# GEMM Tiling Constraints - Snitch

> **File:** `Deeploy/Targets/Snitch/TileConstraints/GemmTileConstraint.py`
> **Class:** `GemmTileConstraint`
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

This constraint class defines tiling rules for General Matrix Multiplication (GEMM) operations on the Snitch platform. The GEMM operation computes `Y = A * B + C` where A, B, and C are matrices. The constraint supports both transposed and non-transposed inputs and handles optional bias addition.

### Supported Data Types

| Input A Type | Input B Type | Output Y Type |
|--------------|--------------|---------------|
| `float32_t`  | `float32_t`  | `float32_t`   |

### Parent Class

- **Extends:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `A` | `[..., M, N]` or `[..., N, M]` | Row-major (depends on transA) | First input matrix |
| `B` | `[..., N, O]` or `[..., O, N]` | Row-major (depends on transB) | Second input matrix |
| `C` | `[..., M, O]` | Row-major (optional) | Bias matrix |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[..., M, O]` | Row-major | Output matrix (Y = A * B + C) |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `A` | `str` | Name of first input matrix buffer |
| `B` | `str` | Name of second input matrix buffer |
| `C` | `str` | Name of bias matrix buffer (optional) |
| `data_out` | `str` | Name of output buffer |
| `transA` | `int` | Whether to transpose A (0=no, 1=yes) |
| `transB` | `int` | Whether to transpose B (0=no, 1=yes) |
| `M` | `int` | Output height dimension |
| `N` | `int` | Inner/reduction dimension |
| `O` | `int` | Output width dimension |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions that must hold for correct matrix multiplication.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `AHeightDimVar` | `A` | dimCount-2 or dimCount-1 | Height of A (M or N depending on transA) |
| `AWidthDimVar` | `A` | dimCount-1 or dimCount-2 | Width of A (N or M depending on transA) |
| `BHeightDimVar` | `B` | dimCount-2 or dimCount-1 | Height of B (N or O depending on transB) |
| `BWidthDimVar` | `B` | dimCount-1 or dimCount-2 | Width of B (O or N depending on transB) |
| `YHeightDimVar` | `data_out` | dimCount-2 | Output height (M) |
| `YWidthDimVar` | `data_out` | dimCount-1 | Output width (O) |
| `CHeightDimVar` | `C` | dimCount-2 | Bias height (M) (if C exists) |
| `CWidthDimVar` | `C` | dimCount-1 | Bias width (O) (if C exists) |

### Constraints

#### Output Height Constraint

**Proposition:**
```
YHeightDimVar == AHeightDimVar
```

**Description:** The output height (M) must equal the height dimension of matrix A (considering transposition).

---

#### Output Width Constraint

**Proposition:**
```
YWidthDimVar == BWidthDimVar
```

**Description:** The output width (O) must equal the width dimension of matrix B (considering transposition).

---

#### Inner Dimension Constraint

**Proposition:**
```
AWidthDimVar == BHeightDimVar
```

**Description:** The inner/reduction dimension (N) must match between A's width and B's height for valid matrix multiplication.

---

#### Bias Dimension Constraints (if C exists)

**Propositions:**
```
CHeightDimVar == YHeightDimVar
CWidthDimVar == YWidthDimVar
```

**Description:** If a bias matrix C is provided, its dimensions must match the output dimensions exactly.

---

## Policy Constraints

Policy constraints define optimization hints and platform-specific restrictions for efficient tiling.

### Constraints

#### Full Inner Dimension Constraint

**Proposition:**
```
AWidthDimVar == AWidthDimVar.Max()
```

**Rationale:** The inner dimension (N) must not be tiled. This is required because partial reduction along the inner dimension would produce incorrect results without accumulation logic.

**Priority:** Hard constraint (must be satisfied)

---

#### Output Height Divisibility Constraint

**Proposition:**
```
YHeightDimVar % 8 == 0  (when M > 8)
```

**Rationale:** The output height dimension (M) is parallelized across the 8 cores in the Snitch cluster. Keeping M divisible by 8 ensures balanced workload distribution.

**Priority:** Performance hint (priority = 1)

---

## Tiling Schedule

### Input Cube Computation

The input cubes for A and B are computed based on the output tile position, handling transposition:

**Algorithm:**

```python
for each output tile YCube:
    MOffset, OOffset = YCube.offset[-2:]
    MSize, OSize = YCube.dims[-2:]

    # Handle batch dimension
    if 3D tensor:
        BatchOffset = YCube.offset[0]
        BatchSize = YCube.dims[0]
    else:
        BatchOffset = 0
        BatchSize = 1

    # Inner dimension is always full
    NOffset = 0
    NSize = N (full)

    # Compute A cube based on transposition
    if transA == 0:
        ACube = HyperRectangle((BatchOffset, MOffset, NOffset), (BatchSize, MSize, NSize))
    else:
        ACube = HyperRectangle((BatchOffset, NOffset, MOffset), (BatchSize, NSize, MSize))

    # Compute B cube based on transposition
    if transB == 0:
        BCube = HyperRectangle((BatchOffset, NOffset, OOffset), (BatchSize, NSize, OSize))
    else:
        BCube = HyperRectangle((BatchOffset, OOffset, NOffset), (BatchSize, OSize, NSize))
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"A": ACube, "B": BCube, "C": YCube},
    ...
]

outputLoadSchedule = [
    {"data_out": YCube},
    ...
]
```

---

## Replacements

Variables that are replaced with tile-specific values during code generation.

| Variable | Type | Description |
|----------|------|-------------|
| `M` | `uint32_t*` | Tiled output height dimension |
| `O` | `uint32_t*` | Tiled output width dimension |
| `batch` | `uint32_t*` | Batch size for current tile |

---

## Notes and Limitations

### Known Limitations

1. **No inner dimension tiling:** The reduction dimension (N) cannot be tiled due to accumulation requirements. The entire inner dimension must fit in memory.

2. **Batch dimension support:** Only 2D and 3D tensors are supported. The batch dimension is in index 0 for 3D tensors.

3. **Bias must match output:** If bias C is provided, it must have the same tiled dimensions as the output.

### Platform-Specific Notes

- **Parallelization:** The Snitch cluster has 8 cores. Work is parallelized across the M (output height) dimension.
- **Memory alignment:** For optimal performance, keep M divisible by 8 for balanced core utilization.

### Performance Considerations

1. **Maximize M tile size** while respecting memory constraints to improve core utilization
2. **Keep M divisible by 8** to balance workload across cores
3. **Inner dimension must be full** - ensure N fits entirely in L1 memory

---

## References

- Source file: `Deeploy/Targets/Snitch/TileConstraints/GemmTileConstraint.py`
- Related templates: `Deeploy/Targets/Snitch/Templates/GemmTemplate.py`
- Tiler configuration: `Deeploy/Targets/Snitch/Tiler.py`
