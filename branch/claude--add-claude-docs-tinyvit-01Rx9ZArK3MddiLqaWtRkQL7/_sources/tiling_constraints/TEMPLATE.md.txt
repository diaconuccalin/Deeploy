# Tiling Constraint Documentation Template

> **How to use this template:**
> 1. Copy this file to create documentation for a specific operator/platform combination
> 2. Name it: `<OperatorName>_<Platform>.md` (e.g., `Conv2D_PULPOpen.md`)
> 3. Fill in all sections with the appropriate information from the TileConstraint source code
> 4. Remove this instruction block and any unused optional sections

---

# [Operator Name] Tiling Constraints - [Platform]

> **File:** `Deeploy/Targets/[Platform]/TileConstraints/[ConstraintFile].py`
> **Class:** `[ConstraintClassName]`
> **Last Updated:** YYYY-MM-DD

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

[Brief description of the operator and what this tiling constraint accomplishes]

### Supported Data Types

| Input Type | Weight Type | Output Type |
|------------|-------------|-------------|
| `int8_t`   | `int8_t`    | `int32_t`   |
| `float32_t`| `float32_t` | `float32_t` |

### Parent Class

- **Inherits from:** `[ParentClass]` (if applicable)
- **Extends:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[N, H, W, C]` | NHWC | Input activation tensor |
| `weight` | `[C_out, K_H, K_W, C_in]` | OIHW | Weight/kernel tensor |
| `bias` | `[C_out]` | C | Bias tensor (optional) |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[N, H_out, W_out, C_out]` | NHWC | Output activation tensor |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `strides` | `List[int]` | Convolution strides `[stride_h, stride_w]` |
| `pads` | `List[int]` | Padding `[pad_top, pad_left, pad_bottom, pad_right]` |
| `dilations` | `List[int]` | Dilation factors `[dilation_h, dilation_w]` |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions that must hold for correct operation.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `inputBatchVar` | `data_in` | 0 | Batch size |
| `inputHeightVar` | `data_in` | 1 | Input height |
| `inputWidthVar` | `data_in` | 2 | Input width |
| `inputChannelVar` | `data_in` | 3 | Input channels |
| `outputBatchVar` | `data_out` | 0 | Output batch size |
| `outputHeightVar` | `data_out` | 1 | Output height |
| `outputWidthVar` | `data_out` | 2 | Output width |
| `outputChannelVar` | `data_out` | 3 | Output channels |

### Constraints

#### Batch Dimension Constraint

**Proposition:**
```
outputBatchVar == inputBatchVar
```

**Description:** The batch dimension must be equal between input and output tensors.

---

#### Output Height Constraint

**Proposition:**
```
outputHeightVar == (effectiveHeight - dilations[0] * (weightHeightVar - 1) - 1) // strides[0] + 1
```

**Where:**
```
effectiveHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1]))
                  - ((weightHeightVar - 1) * (inputHeightVar != inputBuffer.shape[1]))
```

**Description:**
- When the input tile is the full tensor size, padding is added to the effective height
- When the input tile is smaller (tiled), the kernel overlap is subtracted to account for worst-case memory allocation
- The output height is computed using the standard convolution output size formula

---

#### Output Width Constraint

**Proposition:**
```
outputWidthVar == (effectiveWidth - dilations[1] * (weightWidthVar - 1) - 1) // strides[1] + 1
```

**Where:**
```
effectiveWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[2]))
                 - ((weightWidthVar - 1) * (inputWidthVar != inputBuffer.shape[2]))
```

**Description:** Same logic as height constraint, applied to width dimension.

---

#### Channel Dimension Constraint

**Proposition:**
```
inputChannelVar == (weightInChannelVar * group)
```

**Description:** The input channels must match the weight input channels multiplied by the group count.

---

#### Weight-Output Channel Constraint

**Proposition:**
```
weightOutChannelVar == outputChannelVar
```

**Description:** Weight output channels must equal the output tensor channels.

---

## Policy Constraints

Policy constraints define optimization hints, algorithm requirements, and platform-specific restrictions.

### Constraints

#### Full Input Channel Constraint

**Proposition:**
```
inputChannelVar == parseDict['ch_im_in']
```

**Rationale:** Required for im2col algorithm - prevents partial channel results.

**Priority:** Hard constraint (must be satisfied)

---

#### Minimum Input Height Constraint

**Proposition:**
```
effectiveInputHeight >= parseDict['dim_kernel_x']
```

**Rationale:** Input tile must be at least kernel size to apply convolution.

**Priority:** Hard constraint

---

#### Minimum Input Width Constraint

**Proposition:**
```
effectiveInputWidth >= parseDict['dim_kernel_y']
```

**Rationale:** Input tile must be at least kernel size to apply convolution.

**Priority:** Hard constraint

---

#### Stride Alignment Constraint (Height)

**Proposition:**
```
(effectiveInputHeight % strides[0]) == 0
```

**Rationale:** Ensures input tiles are compatible with stride for correct output computation.

**Priority:** Hard constraint

---

#### Stride Alignment Constraint (Width)

**Proposition:**
```
(effectiveInputWidth % strides[1]) == 0
```

**Rationale:** Ensures input tiles are compatible with stride for correct output computation.

**Priority:** Hard constraint

---

#### Weight Non-Tiling Constraints

**Propositions:**
```
weightHeightVar == parseDict['dim_kernel_x']
weightWidthVar == parseDict['dim_kernel_y']
weightInChannelVar * parseDict['group'] == parseDict['ch_im_in']
```

**Rationale:** Weights should not be tiled - full kernel required for each tile computation.

**Priority:** Hard constraint

---

#### Output Channel Divisibility (Optional)

**Proposition:**
```
outputChannelVar % 8 == 0  (when ch_im_out >= 8)
```

**Rationale:** Performance optimization for SIMD operations.

**Priority:** Performance hint (priority = 1)

---

## Tiling Schedule

### Input Cube Computation

The input cube for each output tile is computed based on the output tile position and the convolution parameters.

**Algorithm:**

```python
# Compute actual tile padding based on tile position
tilePadTop = padTop if (outputHAbsoluteOffset == 0) else 0
tilePadLeft = padLeft if (outputWAbsoluteOffset == 0) else 0
tilePadBottom = padBottom if (outputHAbsoluteOffset + outputHSize == outputDims[1]) else 0
tilePadRight = padRight if (outputWAbsoluteOffset + outputWSize == outputDims[2]) else 0

# Compute input offset (relative to upstream tile)
inputHOffset = max(outputHOffset * strideH - padTop, 0)
inputWOffset = max(outputWOffset * strideW - padLeft, 0)

# Compute input dimensions
inputHSize = outputHSize * strideH + (kernelH - 1) - (tilePadTop + tilePadBottom)
inputWSize = outputWSize * strideW + (kernelW - 1) - (tilePadLeft + tilePadRight)
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": InCube, "weight": WeightCube, "bias": BiasCube},
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
| `ch_im_in` | `uint16_t` | Input channels (full) |
| `ch_im_out` | `uint16_t` | Tiled output channels |
| `padding_y_top` | `uint8_t` | Top padding for tile |
| `padding_y_bottom` | `uint8_t` | Bottom padding for tile |
| `padding_x_left` | `uint8_t` | Left padding for tile |
| `padding_x_right` | `uint8_t` | Right padding for tile |

---

## Visual Explanation

### Tiling Strategy

```
Input Tensor [N, H, W, C]
┌─────────────────────────────────┐
│                                 │
│   ┌─────────┐                   │
│   │ Tile 0  │  ← Full channels  │
│   └─────────┘                   │
│        ↓                        │
│   ┌─────────┐                   │
│   │ Tile 1  │  ← Overlap with   │
│   └─────────┘    Tile 0         │
│                                 │
└─────────────────────────────────┘

Output Tensor [N, H_out, W_out, C_out]
┌─────────────────────────────────┐
│   ┌─────┐                       │
│   │ T0  │                       │
│   └─────┘                       │
│   ┌─────┐                       │
│   │ T1  │  ← No overlap         │
│   └─────┘                       │
└─────────────────────────────────┘
```

### Padding Handling

```
Full Tensor Case:
┌───────────────────┐
│  pad  │     │ pad │
│───────┼─────┼─────│
│       │     │     │
│  pad  │Input│ pad │
│       │     │     │
│───────┼─────┼─────│
│  pad  │     │ pad │
└───────────────────┘

Tiled Case (Middle Tile):
┌─────────┐
│         │  ← No padding
│  Input  │    (internal tile)
│  Tile   │
│         │
└─────────┘
```

---

## Examples

### Example 1: Basic 3x3 Convolution Tiling

**Input Parameters:**
- Input shape: `[1, 32, 32, 64]`
- Weight shape: `[128, 3, 3, 64]`
- Strides: `[1, 1]`
- Padding: `[1, 1, 1, 1]`
- L1 Memory: 32KB

**Resulting Tiles:**
- Output tile 0: Shape `[1, 16, 32, 128]`, Offset `[0, 0, 0, 0]`
- Output tile 1: Shape `[1, 16, 32, 128]`, Offset `[0, 16, 0, 0]`

**Input Tiles (with overlap):**
- Input tile 0: Shape `[1, 18, 32, 64]`, Offset `[0, 0, 0, 0]`, Padding `[1, 0, 0, 1]`
- Input tile 1: Shape `[1, 18, 32, 64]`, Offset `[0, 15, 0, 0]`, Padding `[0, 1, 0, 1]`

---

## Notes and Limitations

### Known Limitations

1. **No channel tiling for depthwise convolutions:** Due to kernel implementation requirements, depthwise convolutions must process all channels in a single tile.

2. **Minimum tile size:** Output tiles must be at least 1x1 to produce valid results.

3. **Weight memory:** Weights are not tiled - entire kernel must fit in L1.

### Platform-Specific Notes

- **Memory alignment:** On PULP platforms, tensors should be 4-byte aligned for optimal DMA performance.
- **Parallelization:** Output channel dimension is parallelized across cores when possible.

### Performance Considerations

1. **Prefer larger output tiles** to minimize redundant input data transfer (kernel overlap)
2. **Keep output channels divisible by 8** for optimal SIMD utilization
3. **Consider memory bandwidth** when choosing tile sizes

---

## References

- Source file: `Deeploy/Targets/[Platform]/TileConstraints/[ConstraintFile].py`
- Related templates: `Deeploy/Targets/[Platform]/Templates/[TemplateFile].py`
- Tiler configuration: `Deeploy/Targets/[Platform]/Tiler.py`
