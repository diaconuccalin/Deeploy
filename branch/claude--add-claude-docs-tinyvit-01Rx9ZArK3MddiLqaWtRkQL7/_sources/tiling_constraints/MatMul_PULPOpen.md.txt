# MatMul Tiling Constraints - PULPOpen

> **File:** `Deeploy/Targets/PULPOpen/TileConstraints/MatMulTileConstraint.py`
> **Class:** `MatMulTileConstraint`
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

This file defines tiling constraints for Matrix Multiplication (MatMul) operations on the PULPOpen platform. The constraint handles batched matrix multiplication with support for transposed inputs and broadcasting.

Matrix multiplication computes: `C = A @ B`

Where:
- A has shape `[..., M, N]`
- B has shape `[..., N, O]`
- C has shape `[..., M, O]`

The tiling strategy allows tiling on M and O dimensions but requires the full N dimension (inner/reduction dimension) to avoid partial results.

### Supported Data Types

| Input Type | Weight Type | Output Type |
|------------|-------------|-------------|
| `int8_t`   | `int8_t`    | `int32_t`   |

### Parent Class

- **Inherits from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `A` | `[..., M, N]` | Row-major | First input matrix |
| `B` | `[..., N, O]` | Row-major | Second input matrix |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[..., M, O]` | Row-major | Output matrix |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `A` | `str` | Name of first input buffer |
| `B` | `str` | Name of second input buffer |
| `data_out` | `str` | Name of output buffer |
| `transA` | `int` | Transpose flag for A (0 or 1) |
| `transB` | `int` | Transpose flag for B (0 or 1) |
| `M` | `int` | First dimension of output (rows of A) |
| `N` | `int` | Inner/reduction dimension |
| `O` | `int` | Second dimension of output (cols of B) |

---

## Geometrical Constraints

### Dimension Variables

The constraint handles arbitrary batch dimensions plus the 2D matrix dimensions.

| Variable | Tensor | Dimension | Description |
|----------|--------|-----------|-------------|
| `AMatrixFirstDimVar` | `A` | -2 (adjusted for transA) | Rows of A (M) |
| `AMatrixSecondDimVar` | `A` | -1 (adjusted for transA) | Cols of A (N) |
| `BMatrixFirstDimVar` | `B` | -2 (adjusted for transB) | Rows of B (N) |
| `BMatrixSecondDimVar` | `B` | -1 (adjusted for transB) | Cols of B (O) |
| `outputMatrixFirstDimVar` | `data_out` | -2 | Rows of output (M) |
| `outputMatrixSecondDimVar` | `data_out` | -1 | Cols of output (O) |

### Constraints

#### Batch Dimension Constraints

**Proposition:**
```python
for idx in range(tensorsShapeLenA - 2):
    outputDimVar[tensorsShapeLenOutput - 3 - idx] == ADimVar[tensorsShapeLenA - 3 - idx]

for idx in range(tensorsShapeLenB - 2):
    outputDimVar[tensorsShapeLenOutput - 3 - idx] == BDimVar[tensorsShapeLenB - 3 - idx]
```

**Condition:** Applied when `bufferA.shape[:-2] == bufferB.shape[:-2]`

**Description:** All batch dimensions must match between A, B, and output. This handles multi-dimensional batch scenarios (e.g., `[B1, B2, M, N]`).

---

#### Output Row Dimension Constraint

**Proposition:**
```
outputMatrixFirstDimVar == AMatrixFirstDimVar
```

**Description:** The output row dimension (M) equals the row dimension of A (accounting for transpose).

---

#### Output Column Dimension Constraint

**Proposition:**
```
outputMatrixSecondDimVar == BMatrixSecondDimVar
```

**Description:** The output column dimension (O) equals the column dimension of B (accounting for transpose).

---

#### Inner Dimension Constraint

**Proposition:**
```
AMatrixSecondDimVar == BMatrixFirstDimVar
```

**Description:** The inner/reduction dimension N must match between A and B. This is the fundamental requirement for matrix multiplication: cols(A) == rows(B).

---

## Policy Constraints

### Full Inner Dimension Constraint

**Propositions:**
```
ASecondDimVar == parseDict['N']
BFirstDimVar == parseDict['N']
```

**Rationale:** The inner dimension N cannot be tiled because:
1. It would require accumulating partial results between kernel calls
2. The kernel implementation expects the complete reduction dimension
3. This avoids intermediate buffer management for partial sums

**Priority:** Hard constraint

---

## Tiling Schedule

### Output Cube Processing

For each output cube, corresponding input cubes for A and B are computed:

```python
for cube in outputCubes:
    # Extract M and O dimensions from output
    MOffset, OOffset = cube.offset[-2:]
    MSize, OSize = cube.dims[-2:]

    # Batch handling
    if len(cube.offset) > 2:
        BatchSize = math.prod(cube.dims[:-2])
    else:
        BatchSize = 1

    # A cube: needs full N dimension
    ACube = HyperRectangle(
        (*ABatchOffsets, MOffset, NOffset=0),
        (*ABatchShape, MSize, NSize)
    )

    # B cube: needs full N dimension
    BCube = HyperRectangle(
        (*BBatchOffsets, NOffset=0, OOffset),
        (*BBatchShape, NSize, OSize)
    )
```

### Broadcasting Handling

The constraint properly handles broadcasting when batch dimensions differ:

```python
for idx in range(tensorsShapeLenA - 2):
    if buffA.shape[idx] == buffOut.shape[idx]:
        # Use output cube offset/dims
        ABatchOffsets.append(cube.offset[idx])
        ABatchShape.append(cube.dims[idx])
    else:
        # Broadcasting: keep at 0 offset, size 1
        ABatchOffsets.append(0)
        ABatchShape.append(1)
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"A": ACube, "B": BCube},
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
| `M` | `int8_t*` | Tiled row dimension of output |
| `N` | `int8_t*` | Inner dimension (always full size) |
| `O` | `int8_t*` | Tiled column dimension of output |
| `batch` | `int8_t*` | Product of all batch dimensions |

**Note:** The replacement types use `int8_t*` (pointer class) as a placeholder for the actual pointer type used in code generation.

---

## Visual Explanation

### Matrix Multiplication Tiling

```
A [M, N]              B [N, O]           C [M, O]
+----------+         +----------+       +----------+
|          |         |          |       |          |
|  M rows  |    @    |  N rows  |   =   |  M rows  |
|          |         |          |       |          |
+----------+         +----------+       +----------+
  N cols               O cols             O cols

Tiling Strategy:
- M dimension: CAN be tiled
- N dimension: CANNOT be tiled (reduction)
- O dimension: CAN be tiled
```

### Tiled Computation

```
Output Tiles:
+-------+-------+
|  T0   |  T1   |  <- Tiled on O
+-------+-------+
|  T2   |  T3   |  <- Tiled on M
+-------+-------+

For Output Tile T0:
- Need rows [0:M/2] of A (full N columns)
- Need columns [0:O/2] of B (full N rows)
```

### Batched MatMul

```
Input A: [B, M, N]     Input B: [B, N, O]
          |                     |
          v                     v
   +-------------+       +-------------+
   | Batch 0     |       | Batch 0     |
   | [M, N]      |   @   | [N, O]      |
   +-------------+       +-------------+
   | Batch 1     |       | Batch 1     |
   | [M, N]      |   @   | [N, O]      |
   +-------------+       +-------------+
          |                     |
          v                     v
         Output: [B, M, O]
```

---

## Examples

### Example 1: Simple 2D MatMul

**Input Parameters:**
- A shape: `[128, 64]` (M=128, N=64)
- B shape: `[64, 256]` (N=64, O=256)
- Output shape: `[128, 256]`
- L1 Memory: 32KB

**Tiling Result:**

Since N=64 cannot be tiled, only M and O can be split:

- Output tile 0: Shape `[64, 128]`, Offset `[0, 0]`
- Output tile 1: Shape `[64, 128]`, Offset `[0, 128]`
- Output tile 2: Shape `[64, 128]`, Offset `[64, 0]`
- Output tile 3: Shape `[64, 128]`, Offset `[64, 128]`

**Input Cubes:**
- For tile 0: A `[64, 64]` offset `[0, 0]`, B `[64, 128]` offset `[0, 0]`
- For tile 1: A `[64, 64]` offset `[0, 0]`, B `[64, 128]` offset `[0, 128]`
- etc.

---

### Example 2: Batched MatMul

**Input Parameters:**
- A shape: `[4, 32, 64]` (batch=4, M=32, N=64)
- B shape: `[4, 64, 128]` (batch=4, N=64, O=128)
- Output shape: `[4, 32, 128]`

**Tiling Strategy:**

Batch dimension can also be tiled along with M and O:

- Output tile: Shape `[2, 32, 64]`, Offset `[0, 0, 0]`
- Input A: Shape `[2, 32, 64]`, Offset `[0, 0, 0]`
- Input B: Shape `[2, 64, 64]`, Offset `[0, 0, 0]`

---

### Example 3: Transposed MatMul

**Input Parameters:**
- A shape: `[64, 128]` with transA=1 (effectively `[128, 64]`)
- B shape: `[256, 64]` with transB=1 (effectively `[64, 256]`)
- Result: M=128, N=64, O=256

**Note:** The transpose flags adjust which dimension index is used for M, N, O in the constraint logic.

---

## Notes and Limitations

### Known Limitations

1. **No inner dimension tiling:** The N (reduction) dimension cannot be tiled. This means:
   - For large N values, significant memory is required
   - Memory pressure scales with N dimension

2. **Batch dimension tiling restriction:** Only the last batch dimension can be tiled if there are multiple batch dimensions. All other batch dimensions must be at offset 0.

3. **Replacement type limitation:** All replacement types are `int8_t*` which may need adjustment for different data types.

### Platform-Specific Notes

- **Memory layout:** Matrices are stored in row-major order
- **Alignment:** For optimal performance, dimensions should be aligned to SIMD width (typically 4 or 8 bytes)

### Performance Considerations

1. **Maximize tile reuse:** When tiling both M and O, choose tile sizes that maximize data reuse:
   - Larger M tiles reuse B matrix data
   - Larger O tiles reuse A matrix data

2. **Memory bandwidth:** MatMul is compute-bound for large matrices but can become memory-bound for small tiles. Choose tile sizes that balance compute and memory transfer.

3. **Inner dimension impact:** Since N cannot be tiled, ensure L1 can accommodate:
   - A tile: M_tile * N
   - B tile: N * O_tile
   - Output tile: M_tile * O_tile

4. **Avoid partial results:** The no-tiling constraint on N avoids the need for:
   - Intermediate accumulation buffers
   - Additional synchronization
   - Multiple passes over output

---

## References

- Source file: `Deeploy/Targets/PULPOpen/TileConstraints/MatMulTileConstraint.py`
- Related templates: `Deeploy/Targets/PULPOpen/Templates/`
- Tiler configuration: `Deeploy/Targets/PULPOpen/Tiler.py`
