# Concat Tiling Constraints - Generic

> **File:** `Deeploy/Targets/Generic/TileConstraints/ConcatTileConstraint.py`
> **Class:** `ConcatTileConstraint`
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

The `ConcatTileConstraint` handles the concatenation of two input tensors along a specified axis. The constraint ensures that dimensions match appropriately and enforces that tiling does not split the concatenation axis or any subsequent dimensions, as this would complicate the interleaved memory layout.

### Supported Operations

- ONNX Concat operator (limited to 2 inputs in this implementation)

### Parent Class

- **Inherits from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_in_1` | `[D_0, ..., D_axis, ..., D_n]` | First input tensor |
| `data_in_2` | `[D_0, ..., D_axis', ..., D_n]` | Second input tensor (differs only at axis) |

### Output Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_out` | `[D_0, ..., D_axis + D_axis', ..., D_n]` | Concatenated output tensor |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in_1` | `str` | Name of first input buffer in context |
| `data_in_2` | `str` | Name of second input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `axis` | `int` | Axis along which to concatenate (supports negative indexing) |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions that must hold for correct concatenation.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `inputDim1Var[i]` | `data_in_1` | i | Dimension i of first input |
| `inputDim2Var[i]` | `data_in_2` | i | Dimension i of second input |
| `outputDimVar[i]` | `data_out` | i | Dimension i of output |

### Constraints

#### Non-Axis Dimension Equality

**Proposition:**
```
For all i != posAxis:
    inputDim1Var[i] == outputDimVar[i]
    inputDim2Var[i] == inputDim1Var[i]
```

**Description:** All dimensions except the concatenation axis must be equal across both inputs and the output.

---

#### Concatenation Axis Constraint

**Proposition:**
```
inputDim1Var[posAxis] + inputDim2Var[posAxis] == outputDimVar[posAxis]
```

**Description:** The output dimension at the concatenation axis equals the sum of the two input dimensions at that axis.

---

#### No-Tiling Constraint for Axis and Subsequent Dimensions

**Proposition:**
```
For all i >= posAxis:
    inputDim1Var[i] == input1Shape[i]  (full dimension, no tiling)
    outputDimVar[i] == outputShape[i]   (full dimension, no tiling)
```

**Description:** Dimensions from the concatenation axis onward cannot be tiled. This ensures that the memory layout remains contiguous and the concatenation can be performed correctly without complex index remapping.

---

## Policy Constraints

**Implicit in Geometrical Constraints.** The no-tiling constraint for axis and subsequent dimensions serves as both a geometrical and policy constraint. The rationale is:

1. **Memory contiguity:** Tiling along the concatenation axis would split the data that needs to be interleaved.

2. **Transfer efficiency:** The implementation uses contiguous memory transfers for the elements after the axis.

---

## Tiling Schedule

### Input Cube Computation

For each output cube, compute the corresponding input cubes:

**Algorithm:**
```python
for output_cube in output_cubes:
    in1_cube = copy(output_cube)
    in2_cube = copy(output_cube)

    # Replace concatenation axis dimension with original input sizes
    in1_cube.dims[posAxis:] = (in1Shape[posAxis], *in1_cube.dims[posAxis+1:])
    in2_cube.dims[posAxis:] = (in2Shape[posAxis], *in2_cube.dims[posAxis+1:])
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in_1": In1Cube, "data_in_2": In2Cube},
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
| `iterations` | `uint16_t*` | Number of iterations (product of dims before axis) |
| `in1TransferLength` | `uint16_t*` | Bytes to transfer per iteration from input 1 |
| `in2TransferLength` | `uint16_t*` | Bytes to transfer per iteration from input 2 |

### Replacement Computation

```python
for cube in outputCubes:
    iterations = np.prod(cube.dims[:axis])

    in1TransferLength = np.prod(in1Cube.dims[posAxis:]) * elementSize_in1
    in2TransferLength = np.prod(in2Cube.dims[posAxis:]) * elementSize_in2
```

---

## Visual Explanation

### Concatenation Along Axis 1 (Height)

```
Input 1 [1, 4, 8, 3]          Input 2 [1, 2, 8, 3]
┌─────────────────┐           ┌─────────────┐
│                 │           │             │
│    H=4          │           │    H=2      │
│                 │           │             │
└─────────────────┘           └─────────────┘
          │                          │
          └──────────┬───────────────┘
                     ↓
           Output [1, 6, 8, 3]
           ┌─────────────────┐
           │                 │
           │    H=6          │
           │  (4 + 2)        │
           └─────────────────┘
```

### Tiling Restriction

```
Allowed Tiling (axis=1):
- Tile along dimension 0 (Batch): OK
- Tile along dimension 1 (Height): NOT ALLOWED (concat axis)
- Tile along dimension 2 (Width): NOT ALLOWED (after axis)
- Tile along dimension 3 (Channel): NOT ALLOWED (after axis)

For axis=1, only dimension 0 can be tiled.
```

---

## Examples

### Example 1: Batch Tiling

**Input Parameters:**
- Input 1 shape: `[8, 32, 32, 64]`
- Input 2 shape: `[8, 16, 32, 64]`
- Axis: 1 (height)
- Output shape: `[8, 48, 32, 64]`

**Tiling (along batch):**
- Output Tile 0: Shape `[4, 48, 32, 64]`, Offset `[0, 0, 0, 0]`
- Output Tile 1: Shape `[4, 48, 32, 64]`, Offset `[4, 0, 0, 0]`

**Input Tiles for Tile 0:**
- Input 1 Tile: Shape `[4, 32, 32, 64]`
- Input 2 Tile: Shape `[4, 16, 32, 64]`

**Replacements for Tile 0:**
- `iterations = 4` (batch dimension)
- `in1TransferLength = 32 * 32 * 64 * sizeof(element)` = 65536 bytes (for int8)
- `in2TransferLength = 16 * 32 * 64 * sizeof(element)` = 32768 bytes (for int8)

---

### Example 2: Channel Concatenation (axis=-1)

**Input Parameters:**
- Input 1 shape: `[1, 16, 16, 32]`
- Input 2 shape: `[1, 16, 16, 64]`
- Axis: -1 (or 3, channel)
- Output shape: `[1, 16, 16, 96]`

**Tiling Restrictions:**
- Cannot tile along any dimension (axis is the last dimension)
- Full tensor must fit in L1

**Replacements:**
- `iterations = 1 * 16 * 16 = 256`
- `in1TransferLength = 32 * sizeof(element)`
- `in2TransferLength = 64 * sizeof(element)`

---

### Example 3: No Tiling Required

**Input Parameters:**
- Input 1 shape: `[1, 8, 8, 16]`
- Input 2 shape: `[1, 4, 8, 16]`
- Axis: 1
- L1 Memory: Sufficient for full tensors

**Result:**
- Single tile covering entire tensor
- `iterations = 1`
- `in1TransferLength = 8 * 8 * 16 * sizeof(element)`
- `in2TransferLength = 4 * 8 * 16 * sizeof(element)`

---

## Notes and Limitations

### Known Limitations

1. **Two inputs only:** This implementation supports exactly 2 input tensors. For more inputs, multiple concat operations must be chained.

2. **No tiling along concat axis:** The concatenation axis and all subsequent dimensions cannot be tiled. This may limit memory efficiency for large tensors.

3. **Memory requirements:** All data from axis onward must fit in L1 for both inputs combined.

### Performance Considerations

1. **Choose concat axis wisely:** Earlier axes (lower indices) allow more tiling flexibility.

2. **Transfer length optimization:** The implementation transfers contiguous chunks per iteration, which is efficient for DMA.

3. **Consider tensor layout:** NHWC layout with channel concatenation (axis=3) allows no tiling, while batch concatenation (axis=0) allows full flexibility.

### Common Use Cases

- Skip connections in CNNs (concatenating feature maps)
- Multi-head attention output concatenation
- Feature pyramid networks

---

## References

- Source file: `Deeploy/Targets/Generic/TileConstraints/ConcatTileConstraint.py`
- Related templates: Concat templates in target-specific Template directories
