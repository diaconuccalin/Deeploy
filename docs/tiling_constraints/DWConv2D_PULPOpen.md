# Depthwise Convolution 2D Tiling Constraints - PULPOpen

> **File:** `Deeploy/Targets/PULPOpen/TileConstraints/DWConvTileConstraint.py`
> **Classes:** `DWConv2DTileConstraint`, `RQDWConv2DTileConstraint`
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

This file defines tiling constraints for Depthwise Convolution 2D operations on the PULPOpen platform. Depthwise convolutions apply separate filters to each input channel (groups = number of channels), which is different from standard convolutions.

The file contains two constraint classes:
- **`RQDWConv2DTileConstraint`**: For requantized depthwise convolutions with mul/add parameters
- **`DWConv2DTileConstraint`**: Standard depthwise convolution that extends `Conv2DTileConstraint`

### Supported Data Types

| Input Type | Weight Type | Output Type | Notes |
|------------|-------------|-------------|-------|
| `int8_t`   | `int8_t`    | `int32_t`   | Requantized version |
| `int8_t`   | `int8_t`    | `int8_t`    | Standard version |

### Parent Classes

- **`RQDWConv2DTileConstraint`:** Extends `TileConstraint`
- **`DWConv2DTileConstraint`:** Extends `Conv2DTileConstraint`

---

## Tensor Definitions

### RQDWConv2DTileConstraint

#### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[N, C, H, W]` | NCHW | Input activation tensor |
| `weight` | `[C_out, K_H, K_W]` | CHW | Depthwise weight tensor |
| `mul` | `[C_out]` | C | Requantization multiplier |
| `add` | `[C_out]` | C | Requantization addend |

#### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[N, H_out, W_out, C_out]` | NHWC | Output activation tensor |

### DWConv2DTileConstraint

Uses the same tensor definitions as `Conv2DTileConstraint` with NHWC layout.

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `weight` | `str` | Name of weight buffer in context |
| `mul` | `str` | Name of multiplier buffer (RQ version) |
| `add` | `str` | Name of addend buffer (RQ version) |
| `strides` | `List[int]` | Convolution strides `[stride_h, stride_w]` |
| `pads` | `List[int]` | Padding `[pad_top, pad_left, pad_bottom, pad_right]` |
| `dilations` | `List[int]` | Dilation factors `[dilation_h, dilation_w]` |
| `ch_im_in` | `int` | Number of input channels |
| `ch_im_out` | `int` | Number of output channels |
| `dim_kernel_x` | `int` | Kernel height |
| `dim_kernel_y` | `int` | Kernel width |

---

## Geometrical Constraints

### RQDWConv2DTileConstraint Dimension Variables

| Variable | Tensor | Dimension Index | Layout | Description |
|----------|--------|-----------------|--------|-------------|
| `inputBatchVar` | `data_in` | 0 | NCHW | Batch size |
| `inputChannelVar` | `data_in` | 1 | NCHW | Input channels |
| `inputHeightVar` | `data_in` | 2 | NCHW | Input height |
| `inputWidthVar` | `data_in` | 3 | NCHW | Input width |
| `weightOutChannelVar` | `weight` | 0 | CHW | Output channels |
| `weightHeightVar` | `weight` | 1 | CHW | Kernel height |
| `weightWidthVar` | `weight` | 2 | CHW | Kernel width |
| `outputBatchVar` | `data_out` | 0 | NHWC | Output batch |
| `outputHeightVar` | `data_out` | 1 | NHWC | Output height |
| `outputWidthVar` | `data_out` | 2 | NHWC | Output width |
| `outputChannelVar` | `data_out` | 3 | NHWC | Output channels |

### Constraints

#### Batch Dimension Constraint

**Proposition:**
```
outputBatchVar == inputBatchVar
```

**Description:** The batch dimension must be equal between input and output tensors.

---

#### Channel Dimension Constraints (Depthwise-specific)

**Propositions:**
```
outputChannelVar == weightOutChannelVar
outputChannelVar == inputChannelVar
outputChannelVar == addChannelVar
outputChannelVar == mulChannelVar
```

**Description:** In depthwise convolution, input channels equal output channels (one filter per channel). The requantization parameters must also match the output channel count.

---

#### Output Height Constraint

**Proposition:**
```
outputHeightVar == (effectiveHeight - (weightHeightVar - 1) - 1) // strides[0] + 1
```

**Where:**
```
effectiveHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[2]))
```

**Description:**
- When the input tile is the full tensor height, padding is added to the effective height
- The output height is computed using the standard convolution output size formula
- Note: NCHW layout means height is at index 2

---

#### Output Width Constraint

**Proposition:**
```
outputWidthVar == (effectiveWidth - (weightWidthVar - 1) - 1) // strides[1] + 1
```

**Where:**
```
effectiveWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[3]))
```

**Description:** Same logic as height constraint, applied to width dimension (index 3 in NCHW).

---

## Policy Constraints

### RQDWConv2DTileConstraint Policy Constraints

#### Input Channel Divisibility (L3 Memory)

**Proposition:**
```
inputChannelVar % 4 == 0  (when data_in memory level contains "L3")
```

**Rationale:** Work around tiling issue with non-word-aligned accesses when transferring from L3 memory.

**Priority:** Hard constraint (conditional)

---

#### Weight Non-Tiling Constraints

**Propositions:**
```
weightHeightVar == parseDict['dim_kernel_x']
weightWidthVar == parseDict['dim_kernel_y']
```

**Rationale:** Weights should not be tiled - full kernel required for each tile computation.

**Priority:** Hard constraint

---

#### Minimum Output Size Constraint

**Propositions:**
```
outputHeightVar >= 1 + max([pads[0], pads[2]])
outputWidthVar >= 1 + max([pads[1], pads[3]])
```

**Rationale:** Constraint the minimum tile size such that we can apply at least one kernel. Accounts for padding - needed for MobileNetv1 and similar networks.

**Priority:** Hard constraint

---

#### Minimum Input Size Constraints (Performance Hints)

**Propositions:**
```
inputHeightVar >= dim_kernel_x + pads[0]
inputHeightVar >= dim_kernel_y + pads[1]
```

**Rationale:** Prefer input tiles that are at least kernel size plus padding for efficient convolution.

**Priority:** Performance hint (priority = 1)

---

#### Stride Alignment Constraints

**Propositions:**
```
(inputHeightVar % strides[0]) == 0
(inputWidthVar % strides[1]) == 0
```

**Rationale:** Ensures input tiles are compatible with stride for correct output computation.

**Priority:** Hard constraint

---

### DWConv2DTileConstraint Policy Constraints

In addition to the `Conv2DTileConstraint` policy constraints:

#### Full Channel Constraint

**Propositions:**
```
inputChannelVar == parseDict['ch_im_in']
outputChannelVar == parseDict['ch_im_out']
```

**Rationale:** Due to kernel implementation requirements, depthwise convolutions must process all channels in a single tile. The current DW kernel does not support channel tiling.

**Priority:** Hard constraint

---

## Tiling Schedule

### Input Cube Computation

The input cube is computed using the `Conv2DTileConstraint.computeInputCube` static method, then transformed to NCHW layout for RQDWConv2DTileConstraint.

**Algorithm for RQDWConv2DTileConstraint:**

```python
# Compute input cube in NHWC format
NHWCInCube, padding_tuple = Conv2DTileConstraint.computeInputCube(
    (weightH, weightW), pads, strides, CSize, outputCube, outputShape
)

# Transform to NCHW layout
NCHWInCube = HyperRectangle(
    (NHWCInCube.offset[0], COffset, NHWCInCube.offset[1], NHWCInCube.offset[2]),
    (NHWCInCube.dims[0], CSize, NHWCInCube.dims[1], NHWCInCube.dims[2])
)

# Requantization cubes (1D, channel dimension only)
RequantCube = HyperRectangle((COffset,), (CSize,))

# Weight cube (3D: C, H, W, 1)
WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, weightH, weightW, 1))
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": NCHWInCube, "weight": WeightCube, "add": RequantCube, "mul": RequantCube},
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
| `dim_im_in_x` | `uint16_t` | Tiled input height |
| `dim_im_in_y` | `uint16_t` | Tiled input width |
| `dim_im_out_x` | `uint16_t` | Tiled output height |
| `dim_im_out_y` | `uint16_t` | Tiled output width |
| `ch_im_in` | `uint16_t` | Tiled input channels (equals output channels for DW) |
| `ch_im_out` | `uint16_t` | Tiled output channels |
| `padding_y_top` | `uint8_t` | Top padding for tile |
| `padding_y_bottom` | `uint8_t` | Bottom padding for tile |
| `padding_x_left` | `uint8_t` | Left padding for tile |
| `padding_x_right` | `uint8_t` | Right padding for tile |

---

## Visual Explanation

### Depthwise Convolution Concept

```
Standard Conv:                    Depthwise Conv:
All channels interact             Each channel filtered separately

Input [N,H,W,C]                   Input [N,H,W,C]
       |                                 |
   [KERNEL]                        [C separate kernels]
   (C_in -> C_out)                 (1 -> 1 per channel)
       |                                 |
Output [N,H',W',C_out]            Output [N,H',W',C]
```

### Tiling Strategy (No Channel Tiling)

```
Input Tensor [N, C, H, W] (NCHW)
+----------------------------------+
|  +--------+                      |
|  | Tile 0 | <- All C channels    |
|  +--------+                      |
|      |                           |
|  +--------+                      |
|  | Tile 1 | <- All C channels    |
|  +--------+    (overlap in H/W)  |
|                                  |
+----------------------------------+

Key: Tiles can only be split spatially (H, W),
     not along channel dimension
```

### Layout Transformation (RQDWConv2DTileConstraint)

```
Input (NCHW) -> Processing -> Output (NHWC)

NCHW:                    NHWC:
[Batch]                  [Batch]
  [Channel]                [Height]
    [Height]                 [Width]
      [Width]                  [Channel]
```

---

## Examples

### Example 1: 3x3 Depthwise Convolution

**Input Parameters:**
- Input shape: `[1, 64, 32, 32]` (NCHW)
- Weight shape: `[64, 3, 3]` (CHW)
- Output shape: `[1, 32, 32, 64]` (NHWC)
- Strides: `[1, 1]`
- Padding: `[1, 1, 1, 1]`
- L1 Memory: 32KB

**Tiling Result:**
Since channels cannot be tiled, only spatial tiling is possible:
- Output tile 0: Shape `[1, 16, 32, 64]`, Offset `[0, 0, 0, 0]`
- Output tile 1: Shape `[1, 16, 32, 64]`, Offset `[0, 16, 0, 0]`

**Input Tiles (with overlap):**
- Input tile 0: Shape `[1, 64, 18, 32]` (NCHW), Offset `[0, 0, 0, 0]`
- Input tile 1: Shape `[1, 64, 18, 32]` (NCHW), Offset `[0, 0, 15, 0]`

---

### Example 2: MobileNet-style Depthwise Convolution

**Input Parameters:**
- Input shape: `[1, 128, 16, 16]` (NCHW)
- Weight shape: `[128, 3, 3]` (CHW)
- Strides: `[2, 2]`
- Padding: `[1, 1, 1, 1]`

**Note:** With stride 2, the output spatial dimensions are halved. Channel tiling is still not allowed due to DW kernel requirements.

---

## Notes and Limitations

### Known Limitations

1. **No channel tiling:** Due to kernel implementation requirements, depthwise convolutions must process all channels in a single tile. This is the most significant limitation.

2. **Layout mismatch:** `RQDWConv2DTileConstraint` handles NCHW input but NHWC output, requiring layout transformation in the serialization.

3. **L3 memory alignment:** When input is in L3 memory, channels must be divisible by 4 to avoid alignment issues.

4. **Weight memory:** Weights are not tiled - entire kernel must fit in L1.

### Platform-Specific Notes

- **Memory alignment:** On PULP platforms, tensors should be 4-byte aligned for optimal DMA performance.
- **Depthwise efficiency:** DW convolutions are memory-bandwidth bound rather than compute-bound, so tiling strategy focuses on minimizing data movement.

### Performance Considerations

1. **Prefer larger spatial tiles** to minimize redundant input data transfer (kernel overlap)
2. **Consider memory bandwidth** when choosing tile sizes - DW convolutions have low arithmetic intensity
3. **L3 memory alignment** constraint (divisible by 4) should be considered when planning channel dimensions

### TODO

- Fix DW kernel to include group info and support channel tiling (see comment in code)

---

## References

- Source file: `Deeploy/Targets/PULPOpen/TileConstraints/DWConvTileConstraint.py`
- Parent class: `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py`
- Related templates: `Deeploy/Targets/PULPOpen/Templates/`
- Tiler configuration: `Deeploy/Targets/PULPOpen/Tiler.py`
