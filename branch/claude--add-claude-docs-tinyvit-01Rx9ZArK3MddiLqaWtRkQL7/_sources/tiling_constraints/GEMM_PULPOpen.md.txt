# GEMM Tiling Constraints - PULPOpen

> **File:** `Deeploy/Targets/PULPOpen/TileConstraints/GEMMTileConstraint.py`
> **Classes:** `GEMMTileConstraint`, `FloatGEMMTileConstraint`
> **Last Updated:** 2025-01-18

---

## Table of Contents

1. [Overview](#overview)
2. [Tensor Definitions](#tensor-definitions)
3. [Geometrical Constraints](#geometrical-constraints)
4. [Policy Constraints](#policy-constraints)
5. [Tiling Schedule](#tiling-schedule)
6. [Replacements](#replacements)
7. [Visual Explanation](#visual-explanation)
8. [Notes and Limitations](#notes-and-limitations)

---

## Overview

### Description

The GEMM (General Matrix Multiply) tiling constraint enables tiling of matrix multiplication operations: `C = alpha * A @ B + beta * C`. It supports both requantized integer and floating-point variants with transposition options for input matrices.

### Supported Variants

| Class | Data Types | Requantization |
|-------|------------|----------------|
| `GEMMTileConstraint` | Int8/Int32 | Yes (mul, add) |
| `FloatGEMMTileConstraint` | Float32 | No |

### Matrix Dimensions Convention

```
A: [batch, M, N] (or [batch, N, M] if transA=1)
B: [batch, N, O] (or [batch, O, N] if transB=1)
C: [batch, M, O]
```

Where:
- **M**: Number of rows in output
- **N**: Inner/reduction dimension
- **O**: Number of columns in output

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `A` | `[..., M, N]` or `[..., N, M]` | Row-major | First input matrix |
| `B` | `[..., N, O]` or `[..., O, N]` | Row-major | Second input matrix |
| `C` | `[O]` or `[M, O]` | Row-major | Bias/accumulator (RQS: add buffer) |
| `mul` | `[O]` | Vector | Requantization multiplier (RQS only) |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[..., M, O]` | Row-major | Output matrix |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `A` | `str` | Name of matrix A buffer |
| `B` | `str` | Name of matrix B buffer |
| `C` | `str` | Name of bias/accumulator buffer |
| `mul` | `str` | Name of requantization multiplier (RQS) |
| `data_out` | `str` | Name of output buffer |
| `M` | `int` | Output rows |
| `N` | `int` | Inner dimension |
| `O` | `int` | Output columns |
| `transA` | `int` | Transpose A (0 or 1) |
| `transB` | `int` | Transpose B (0 or 1) |

---

## Geometrical Constraints

### Dimension Variables

| Variable | Tensor | Description |
|----------|--------|-------------|
| `AFirstDimVar` | A | M dimension (or N if transA) |
| `ASecondDimVar` | A | N dimension (or M if transA) |
| `BFirstDimVar` | B | N dimension (or O if transB) |
| `BSecondDimVar` | B | O dimension (or N if transB) |
| `outputFirstDimVar` | data_out | M dimension |
| `outputSecondDimVar` | data_out | O dimension |
| `mulDimVar` | mul | Requant multiplier size (RQS) |
| `addDimVar` | C | Bias/add buffer size |

### Constraints

#### Output-A Row Constraint

**Proposition:**
```
outputFirstDimVar == AFirstDimVar
```

**Description:** Output M dimension equals A's first effective dimension.

**Source:** Line 48

---

#### Output-B Column Constraint

**Proposition:**
```
outputSecondDimVar == BSecondDimVar
```

**Description:** Output O dimension equals B's second effective dimension.

**Source:** Line 49

---

#### Inner Dimension Constraint

**Proposition:**
```
ASecondDimVar == BFirstDimVar
```

**Description:** A's columns (N) must equal B's rows (N) - standard matrix multiplication requirement.

**Source:** Line 52

---

#### Requantization Dimension Constraints (RQS only)

**Propositions:**
```
outputSecondDimVar == mulDimVar
outputSecondDimVar == addDimVar
```

**Description:** Requantization vectors must match output columns (O dimension).

**Source:** Lines 57-58

---

#### Bias Dimension Constraints (Float only)

**Propositions:**
```
outputFirstDimVar == addDimVar_1
outputSecondDimVar == addDimVar_2
```

**Description:** For float GEMM, bias C has shape [M, O] matching output.

**Source:** Lines 228-229

---

## Policy Constraints

### Inner Dimension Non-Tiling

**Propositions:**
```
ASecondDimVar == parseDict['N']
BFirstDimVar == parseDict['N']
```

**Rationale:** The inner dimension N cannot be tiled to avoid partial accumulation results. Complete inner products must be computed in each tile.

**Priority:** Hard constraint

**Source:** Lines 80-81

---

### Output Column Divisibility

**Proposition (when O >= 16):**
```
BSecondDimVar % 16 == 0
```

**Rationale:** Performance optimization for SIMD vectorization.

**Priority:** Performance hint

**Source:** Lines 83-85

---

## Tiling Schedule

### Cube Computation

For each output tile, reconstruct corresponding input tiles:

```python
for cube in outputCubes:
    MOffset, OOffset = cube.offset[-2:]
    MSize, OSize = cube.dims[-2:]

    # A tile: [batch, M, N] (full N dimension)
    if transA == 0:
        ACube = HyperRectangle((MOffset, NOffset), (MSize, NSize))
    else:
        ACube = HyperRectangle((NOffset, MOffset), (NSize, MSize))

    # B tile: [batch, N, O] (full N dimension)
    if transB == 0:
        BCube = HyperRectangle((NOffset, OOffset), (NSize, OSize))
    else:
        BCube = HyperRectangle((OOffset, NOffset), (OSize, NSize))

    # Requant tiles: just O dimension
    RequantCube = HyperRectangle((OOffset,), (OSize,))
```

### Load Schedule Structure

```python
# RQS GEMM
inputLoadSchedule = [
    {"A": ACube, "B": BCube, "C": AddCube, "mul": MulCube},
    ...
]

# Float GEMM
inputLoadSchedule = [
    {"A": ACube, "B": BCube, "C": CCube},
    ...
]

outputLoadSchedule = [
    {"data_out": OutCube},
    ...
]
```

---

## Replacements

| Variable | Type | Description |
|----------|------|-------------|
| `M` | `uint16_t` | Tiled output rows |
| `N` | `uint16_t` | Inner dimension (always full) |
| `O` | `uint16_t` | Tiled output columns |
| `batch` | `uint8_t` | Batch size for current tile |

---

## Visual Explanation

### GEMM Tiling Strategy

```
Matrix A [M, N]              Matrix B [N, O]
┌─────────────────┐          ┌─────────────────────┐
│  A_tile_0       │          │                     │
│  [M/2, N]       │          │  B_tile_0  B_tile_1 │
├─────────────────┤    @     │  [N, O/2]  [N, O/2] │
│  A_tile_1       │          │                     │
│  [M/2, N]       │          └─────────────────────┘
└─────────────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
              Output C [M, O]
              ┌─────────────────────┐
              │ C_00    │    C_01   │
              │ [M/2,   │  [M/2,    │
              │  O/2]   │   O/2]    │
              ├─────────┼───────────┤
              │ C_10    │    C_11   │
              │ [M/2,   │  [M/2,    │
              │  O/2]   │   O/2]    │
              └─────────────────────┘

Note: N dimension is NOT tiled (full inner products)
```

### Memory Layout with Transposition

```
transA = 0, transB = 0 (default):
A: [M, N] row-major
B: [N, O] row-major

transA = 1:
A: [N, M] row-major → logically [M, N]

transB = 1:
B: [O, N] row-major → logically [N, O]
```

---

## Examples

### Example 1: Standard GEMM

**Configuration:**
- A: `[64, 128]` (M=64, N=128)
- B: `[128, 256]` (N=128, O=256)
- Output: `[64, 256]`
- L1 Memory: 64KB

**Constraints Applied:**
- N = 128 (full, not tiled)
- O % 16 == 0 (256 satisfies)

**Resulting Tiles:**

| Tile | Output | A Tile | B Tile | Mul/Add |
|------|--------|--------|--------|---------|
| 0 | `[32, 128]` @ `[0, 0]` | `[32, 128]` | `[128, 128]` | `[128]` |
| 1 | `[32, 128]` @ `[0, 128]` | `[32, 128]` | `[128, 128]` | `[128]` |
| 2 | `[32, 128]` @ `[32, 0]` | `[32, 128]` | `[128, 128]` | `[128]` |
| 3 | `[32, 128]` @ `[32, 128]` | `[32, 128]` | `[128, 128]` | `[128]` |

### Example 2: Batched GEMM

**Configuration:**
- A: `[4, 32, 64]` (batch=4, M=32, N=64)
- B: `[4, 64, 128]` (batch=4, N=64, O=128)
- Output: `[4, 32, 128]`

**Tiling:** Only last batch dimension and M/O can be tiled. N=64 is kept full.

---

## Notes and Limitations

### Known Limitations

1. **No inner dimension tiling:** The N dimension cannot be tiled to avoid partial sums, which would require additional accumulation logic.

2. **Batch dimension restriction:** Only the last batch dimension can be tiled when there are multiple batch dimensions.

3. **Transposition overhead:** Transposed matrices may have suboptimal memory access patterns.

### Performance Considerations

1. **Output column alignment:** Keep O divisible by 16 for optimal SIMD performance.

2. **Memory reuse:** Larger M tiles improve A matrix reuse; larger O tiles improve B matrix reuse.

3. **Compute intensity:** GEMM has O(M*N*O) compute and O(M*N + N*O + M*O) memory traffic. Larger tiles improve compute/memory ratio.

### Differences: GEMMTileConstraint vs FloatGEMMTileConstraint

| Aspect | GEMMTileConstraint | FloatGEMMTileConstraint |
|--------|-------------------|-------------------------|
| Data types | Int8/Int32 | Float32 |
| Requantization | Yes (mul, add) | No |
| Bias shape | `[O]` | `[M, O]` |
| Use case | Quantized inference | FP training/inference |

---

## References

- Source file: `Deeploy/Targets/PULPOpen/TileConstraints/GEMMTileConstraint.py`
- Templates: `Deeploy/Targets/PULPOpen/Templates/GEMMTemplate.py`
- Tiler configuration: `Deeploy/Targets/PULPOpen/Tiler.py` (lines 55-59)
