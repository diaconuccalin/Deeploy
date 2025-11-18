# Binary Operator (BOP) Tiling Constraints - Generic

> **File:** `Deeploy/Targets/Generic/TileConstraints/BOPTileConstraint.py`
> **Class:** `BOPTileConstraint`
> **Subclasses:** `AddTileConstraint`, `MulTileConstraint`
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

The Binary Operator (BOP) tiling constraint provides a base class for element-wise binary operations such as addition and multiplication. It enforces that both input tensors and the output tensor have identical shapes, enabling straightforward tiling where all three tensors are tiled identically.

### Supported Operations

| Operation | Constraint Class | Data In Names |
|-----------|------------------|---------------|
| Add | `AddTileConstraint` | `data_in_1`, `data_in_2` |
| Mul | `MulTileConstraint` | `A`, `B` |

### Supported Data Types

| Input 1 Type | Input 2 Type | Output Type |
|--------------|--------------|-------------|
| `int8_t` | `int8_t` | `int8_t` |
| `int16_t` | `int16_t` | `int16_t` |
| `int32_t` | `int32_t` | `int32_t` |
| `float32_t` | `float32_t` | `float32_t` |

### Class Hierarchy

```
TileConstraint
└── BOPTileConstraint
    ├── AddTileConstraint
    └── MulTileConstraint
```

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in_1` (or `A`) | `[d0, d1, ..., dn]` | Any | First input tensor |
| `data_in_2` (or `B`) | `[d0, d1, ..., dn]` | Any | Second input tensor |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` (or `C`) | `[d0, d1, ..., dn]` | Any | Output tensor |

### Class Attributes

| Attribute | Default Value | Description |
|-----------|---------------|-------------|
| `dataIn1Name` | `'data_in_1'` | Name key for first input |
| `dataIn2Name` | `'data_in_2'` | Name key for second input |
| `dataOutName` | `'data_out'` | Name key for output |

**MulTileConstraint overrides:**
```python
dataIn1Name = "A"
dataIn2Name = "B"
dataOutName = "C"
```

---

## Geometrical Constraints

### Dimension Variables

For each dimension `dim` in range `[0, len(input1Shape))`:

| Variable | Tensor | Description |
|----------|--------|-------------|
| `inputDim1Var` | `data_in_1` | Dimension size of first input |
| `inputDim2Var` | `data_in_2` | Dimension size of second input |
| `outputDimVar` | `data_out` | Dimension size of output |

### Constraints

#### Input Equality Constraint

**Proposition (for each dimension `dim`):**
```
inputDim1Var == inputDim2Var
```

**Description:** Both input tensors must have identical dimension sizes.

**Source:** Line 43

---

#### Input-Output Equality Constraint

**Proposition (for each dimension `dim`):**
```
inputDim1Var == outputDimVar
```

**Description:** Output tensor dimensions match input tensor dimensions (element-wise operation).

**Source:** Line 44

---

### Combined Constraint Summary

For an N-dimensional tensor:
```
forall dim in [0, N):
    input1[dim] == input2[dim] == output[dim]
```

---

## Policy Constraints

**None defined.**

The BOPTileConstraint does not add any policy constraints, allowing maximum flexibility in tiling decisions. Any dimension can be tiled without restrictions.

---

## Tiling Schedule

### Cube Computation

For binary operations, the input and output cubes are identical:

```python
for cube in outputCubes:
    inputLoadSchedule.append({
        dataIn1Name: cube,
        dataIn2Name: cube
    })
    outputLoadSchedule.append({
        dataOutName: cube
    })
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in_1": cube0, "data_in_2": cube0},
    {"data_in_1": cube1, "data_in_2": cube1},
    ...
]

outputLoadSchedule = [
    {"data_out": cube0},
    {"data_out": cube1},
    ...
]
```

---

## Replacements

| Variable | Type | Description |
|----------|------|-------------|
| `size` | `uint16_t` | Total number of elements in the tile |

**Computation:**
```python
for cube in outputCubes:
    newSize = np.prod(cube.dims)
    replacements["size"].append(newSize)
```

---

## Visual Explanation

### Element-wise Tiling

```
Input 1: [4, 8, 16]          Input 2: [4, 8, 16]
┌────────────────┐           ┌────────────────┐
│  Tile 0        │           │  Tile 0        │
│  [2, 8, 16]    │           │  [2, 8, 16]    │
├────────────────┤     +     ├────────────────┤
│  Tile 1        │    or     │  Tile 1        │
│  [2, 8, 16]    │     *     │  [2, 8, 16]    │
└────────────────┘           └────────────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
              Output: [4, 8, 16]
              ┌────────────────┐
              │  Tile 0        │
              │  [2, 8, 16]    │
              ├────────────────┤
              │  Tile 1        │
              │  [2, 8, 16]    │
              └────────────────┘
```

### Memory Access Pattern

```
For tile i:
  - Load input1[tile_i_offset : tile_i_offset + tile_i_size]
  - Load input2[tile_i_offset : tile_i_offset + tile_i_size]
  - Compute output[j] = input1[j] op input2[j] for j in tile
  - Store output[tile_i_offset : tile_i_offset + tile_i_size]
```

---

## Examples

### Example 1: Vector Addition

**Configuration:**
- Input 1: `[1024]`
- Input 2: `[1024]`
- Output: `[1024]`
- L1 Memory: 4KB

**Tiling (assuming 3 buffers of 1KB each):**

| Tile | Shape | Offset | Size |
|------|-------|--------|------|
| 0 | `[256]` | `[0]` | 256 |
| 1 | `[256]` | `[256]` | 256 |
| 2 | `[256]` | `[512]` | 256 |
| 3 | `[256]` | `[768]` | 256 |

### Example 2: Matrix Element-wise Multiplication

**Configuration:**
- Input A: `[64, 128]`
- Input B: `[64, 128]`
- Output C: `[64, 128]`

**Possible Tiling Strategies:**

1. **Row-wise tiling:**
   - Tile 0: `[32, 128]` at offset `[0, 0]`
   - Tile 1: `[32, 128]` at offset `[32, 0]`

2. **Block tiling:**
   - Tile 0: `[32, 64]` at offset `[0, 0]`
   - Tile 1: `[32, 64]` at offset `[0, 64]`
   - Tile 2: `[32, 64]` at offset `[32, 0]`
   - Tile 3: `[32, 64]` at offset `[32, 64]`

---

## Notes and Limitations

### Known Limitations

1. **No broadcasting support:** Both inputs must have exactly the same shape. Broadcasting (e.g., adding a scalar to a tensor) requires a different constraint class.

2. **Identical tiling:** All three tensors (input1, input2, output) are tiled identically, which may not be optimal for all memory configurations.

### Performance Considerations

1. **Memory bandwidth:** Binary operations are typically memory-bound. Maximize tile size to improve cache utilization.

2. **Vectorization:** Choose tile sizes that align with SIMD vector widths (e.g., multiples of 4 or 8 elements).

3. **DMA efficiency:** Larger contiguous transfers are more efficient than many small transfers.

### Platform Inheritance

The `BOPTileConstraint` is defined in `Deeploy/Targets/Generic/TileConstraints/` and is inherited by all platforms. Platform-specific optimizations (like SIMD alignment) should be implemented in derived classes or policy constraints.

---

## References

- Source file: `Deeploy/Targets/Generic/TileConstraints/BOPTileConstraint.py`
- Add constraint: `Deeploy/Targets/Generic/TileConstraints/AddTileConstraint.py`
- Mul constraint: `Deeploy/Targets/Generic/TileConstraints/MulTileConstraint.py`
