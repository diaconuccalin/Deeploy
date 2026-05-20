# Conv2D Tiling Constraints - PULPOpen

> **File:** `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py`
> **Classes:** `Conv2DTileConstraint`, `RQConv2DTileConstraint`
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
8. [Examples](#examples)
9. [Notes and Limitations](#notes-and-limitations)

---

## Overview

### Description

The Conv2D tiling constraint enables spatial tiling of 2D convolution operations on the PULPOpen platform. It supports both floating-point and requantized (integer) convolutions with proper handling of:
- Padding at tensor boundaries
- Kernel overlap between adjacent tiles
- Stride-based output computation
- Grouped convolutions

### Supported Data Types

| Variant | Input Type | Weight Type | Output Type |
|---------|------------|-------------|-------------|
| Float Conv2D | `float32_t` | `float32_t` | `float32_t` |
| RQS Conv2D | `int8_t` | `int8_t` | `int8_t` / `int32_t` |

### Class Hierarchy

```
TileConstraint
├── Conv2DTileConstraint (Float convolutions)
│   └── DWConv2DTileConstraint (Depthwise float)
└── RQConv2DTileConstraint (Requantized convolutions)
    └── RQDWConv2DTileConstraint (Depthwise requantized)
```

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[N, H, W, C_in]` | NHWC | Input activation tensor |
| `weight` | `[C_out, K_H, K_W, C_in/group]` | OHWI | Weight/kernel tensor |
| `bias` | `[C_out]` | C | Bias tensor (optional) |
| `mul` | `[C_out]` | C | Requantization multiplier (RQS only) |
| `add` | `[C_out]` | C | Requantization offset (RQS only) |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[N, H_out, W_out, C_out]` | NHWC | Output activation tensor |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `weight` | `str` | Name of weight buffer in context |
| `bias` | `str` | Name of bias buffer (or "NULL") |
| `strides` | `List[int]` | Convolution strides `[stride_h, stride_w]` |
| `pads` | `List[int]` | Padding `[pad_top, pad_left, pad_bottom, pad_right]` |
| `dilations` | `List[int]` | Dilation factors `[dilation_h, dilation_w]` |
| `group` | `int` | Number of groups for grouped convolution |
| `ch_im_in` | `int` | Total input channels |
| `ch_im_out` | `int` | Total output channels |
| `dim_kernel_x` | `int` | Kernel height |
| `dim_kernel_y` | `int` | Kernel width |
| `has_bias` | `str` | "true" or "false" |

---

## Geometrical Constraints

### Dimension Variables

| Variable | Tensor | Dim Index | Description |
|----------|--------|-----------|-------------|
| `inputBatchVar` | `data_in` | 0 | Batch size |
| `inputHeightVar` | `data_in` | 1 | Input height |
| `inputWidthVar` | `data_in` | 2 | Input width |
| `inputChannelVar` | `data_in` | 3 | Input channels |
| `weightOutChannelVar` | `weight` | 0 | Output channels |
| `weightHeightVar` | `weight` | 1 | Kernel height |
| `weightWidthVar` | `weight` | 2 | Kernel width |
| `weightInChannelVar` | `weight` | 3 | Input channels per group |
| `outputBatchVar` | `data_out` | 0 | Output batch size |
| `outputHeightVar` | `data_out` | 1 | Output height |
| `outputWidthVar` | `data_out` | 2 | Output width |
| `outputChannelVar` | `data_out` | 3 | Output channels |
| `biasDimVar` | `bias` | 0 | Bias dimension (if has_bias) |

### Constraints

#### 1. Batch Dimension Constraint

**Proposition:**
```
outputBatchVar == inputBatchVar
```

**Description:** Batch dimension is preserved between input and output.

**Source:** Line 305

---

#### 2. Output Height Constraint

**Proposition:**
```
outputHeightVar == (effectiveHeight - dilations[0] * (weightHeightVar - 1) - 1) // strides[0] + 1
```

**Where `effectiveHeight` is defined as:**
```
effectiveHeight = inputHeightVar
    + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1]))
    - ((weightHeightVar - 1) * (inputHeightVar != inputBuffer.shape[1]))
```

**Description:**
- **Full tensor case** (`inputHeightVar == inputBuffer.shape[1]`): Padding is added to compute the output size
- **Tiled case** (`inputHeightVar != inputBuffer.shape[1]`): Kernel overlap (`weightHeightVar - 1`) is subtracted to account for worst-case memory allocation at tile boundaries

This formulation ensures proper memory allocation for tiles while maintaining correct output size computation.

**Source:** Lines 311-312, 316-317

---

#### 3. Output Width Constraint

**Proposition:**
```
outputWidthVar == (effectiveWidth - dilations[1] * (weightWidthVar - 1) - 1) // strides[1] + 1
```

**Where `effectiveWidth` is defined as:**
```
effectiveWidth = inputWidthVar
    + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[2]))
    - ((weightWidthVar - 1) * (inputWidthVar != inputBuffer.shape[2]))
```

**Description:** Same logic as height constraint, applied to width dimension.

**Source:** Lines 313-314, 318-319

---

#### 4. Input Channel Constraint

**Proposition:**
```
inputChannelVar == (weightInChannelVar * group)
```

**Description:** Total input channels must equal weight input channels multiplied by group count.

**Source:** Line 323

---

#### 5. Weight-Output Channel Constraint

**Proposition:**
```
weightOutChannelVar == outputChannelVar
```

**Description:** Weight output channel dimension must match output tensor channels.

**Source:** Line 327

---

#### 6. Bias Dimension Constraint (if has_bias)

**Proposition:**
```
biasDimVar == outputChannelVar
```

**Description:** Bias vector length must match output channels.

**Source:** Line 331

---

## Policy Constraints

### Constraints

#### 1. Full Input Channel Constraint

**Proposition:**
```
inputChannelVar == parseDict['ch_im_in']
```

**Rationale:** The im2col algorithm requires complete input channels to avoid partial accumulation results between kernel calls.

**Priority:** Hard constraint

**Source:** Line 371

---

#### 2. Minimum Effective Input Height Constraint

**Proposition:**
```
effectiveInputHeight >= parseDict['dim_kernel_x']
```

**Where:**
```
effectiveInputHeight = inputHeightVar
    + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1]))
    - ((weightHeightVar - 1) * (inputHeightVar != inputBuffer.shape[1]))
```

**Rationale:** Input tile (including padding and overlap adjustments) must be at least kernel size to apply convolution.

**Priority:** Hard constraint

**Source:** Lines 362-367, 374

---

#### 3. Minimum Effective Input Width Constraint

**Proposition:**
```
effectiveInputWidth >= parseDict['dim_kernel_y']
```

**Rationale:** Same as height - ensures valid convolution application.

**Priority:** Hard constraint

**Source:** Lines 362-367, 375

---

#### 4. Stride Alignment (Height)

**Proposition:**
```
(effectiveInputHeight % strides[0]) == 0
```

**Rationale:** Ensures input tiles produce integer output dimensions when divided by stride.

**Priority:** Hard constraint

**Source:** Line 378

---

#### 5. Stride Alignment (Width)

**Proposition:**
```
(effectiveInputWidth % strides[1]) == 0
```

**Rationale:** Ensures input tiles produce integer output dimensions when divided by stride.

**Priority:** Hard constraint

**Source:** Line 379

---

#### 6. Weight Non-Tiling Constraints

**Propositions:**
```
weightHeightVar == parseDict['dim_kernel_x']
weightWidthVar == parseDict['dim_kernel_y']
weightInChannelVar * parseDict['group'] == parseDict['ch_im_in']
```

**Rationale:** Weights are not tiled - the complete kernel is required for each tile computation.

**Priority:** Hard constraint

**Source:** Lines 382-384

---

## Tiling Schedule

### Input Cube Computation

For each output tile, the corresponding input cube is computed using `Conv2DTileConstraint.computeInputCube()`:

**Algorithm (Lines 409-459):**

```python
def computeInputCube(kernelShape, pads, strides, inputCSize, outputCube,
                     outputDims, inputDims=None, outputAbsoluteOffsets=None):

    # 1. Extract output tile information
    (outputBatchOffset, outputHOffset, outputWOffset, _) = outputCube.offset
    (outputBatchSize, outputHSize, outputWSize, _) = outputCube.dims

    # 2. Compute tile-specific padding (only at tensor edges)
    tilePadTop = padTop if (outputHAbsoluteOffset == 0) else 0
    tilePadLeft = padLeft if (outputWAbsoluteOffset == 0) else 0
    tilePadBottom = padBottom if (outputHAbsoluteOffset + outputHSize == outputDims[1]) else 0
    tilePadRight = padRight if (outputWAbsoluteOffset + outputWSize == outputDims[2]) else 0

    # 3. Compute input offset (relative to upstream tile)
    inputHOffset = max(outputHOffset * strideH - padTop, 0)
    inputWOffset = max(outputWOffset * strideW - padLeft, 0)

    # 4. Compute input dimensions
    inputHSize = outputHSize * strideH + (kernelH - 1) - (tilePadTop + tilePadBottom)
    inputWSize = outputWSize * strideW + (kernelW - 1) - (tilePadLeft + tilePadRight)

    # 5. Clamp to input boundaries (for edge tiles)
    if inputDims is not None:
        inputHSize = min(inputHSize, inputDims[1] - inputHOffset)
        inputWSize = min(inputWSize, inputDims[2] - inputWOffset)

    return InCube, (tilePadLeft, tilePadRight, tilePadTop, tilePadBottom)
```

### Load Schedule Structure

```python
# For Float Conv2D
inputLoadSchedule = [
    {"data_in": InCube, "weight": WeightCube, "bias": BiasCube},
    ...
]

# For RQS Conv2D
inputLoadSchedule = [
    {"data_in": InCube, "weight": WeightCube, "mul": MulCube, "add": AddCube},
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
| `dim_im_in_x` | `uint16_t` | Tiled input height |
| `dim_im_in_y` | `uint16_t` | Tiled input width |
| `dim_im_out_x` | `uint16_t` | Tiled output height |
| `dim_im_out_y` | `uint16_t` | Tiled output width |
| `ch_im_in` | `uint16_t` | Input channels (always full) |
| `ch_im_out` | `uint16_t` | Tiled output channels |
| `padding_y_top` | `uint8_t` | Top padding for current tile |
| `padding_y_bottom` | `uint8_t` | Bottom padding for current tile |
| `padding_x_left` | `uint8_t` | Left padding for current tile |
| `padding_x_right` | `uint8_t` | Right padding for current tile |

---

## Visual Explanation

### Spatial Tiling with Kernel Overlap

```
Input Tensor [1, 8, 8, C]                 Output Tensor [1, 6, 6, C_out]
with 3x3 kernel, stride 1, pad 1

Input:                                    Output:
┌────────────────────────┐                ┌──────────────────┐
│ Tile 0 (with overlap)  │                │     Tile 0       │
│ ┌──────────────────┐   │                │ ┌────────────┐   │
│ │ H=5, W=8         │   │  ────────►     │ │ H=3, W=6   │   │
│ │ (includes K-1=2  │   │                │ └────────────┘   │
│ │  overlap rows)   │   │                │                  │
│ └──────────────────┘   │                │     Tile 1       │
│ ┌──────────────────┐   │                │ ┌────────────┐   │
│ │ Tile 1 (overlap) │   │  ────────►     │ │ H=3, W=6   │   │
│ │ H=5, W=8         │   │                │ └────────────┘   │
│ └──────────────────┘   │                └──────────────────┘
└────────────────────────┘

Note: Input tiles overlap by (kernel_size - 1) = 2 rows
```

### Edge Padding Handling

```
Tile Position and Padding:

┌─────────────────────────────────────┐
│  Top-Left Tile                      │
│  padding: [top, left, 0, 0]         │
│  ┌───────┬─────────┐                │
│  │ PAD   │ PAD     │                │
│  ├───────┼─────────┤                │
│  │ PAD   │  DATA   │                │
│  └───────┴─────────┘                │
│                                     │
│  Middle Tile                        │
│  padding: [0, 0, 0, 0]              │
│  ┌─────────┐                        │
│  │  DATA   │ ◄─ No padding          │
│  └─────────┘                        │
│                                     │
│  Bottom-Right Tile                  │
│  padding: [0, 0, bottom, right]     │
│  ┌─────────┬───────┐                │
│  │  DATA   │ PAD   │                │
│  ├─────────┼───────┤                │
│  │ PAD     │ PAD   │                │
│  └─────────┴───────┘                │
└─────────────────────────────────────┘
```

### Effective Height/Width Computation

```
For Full Tensor:
effectiveHeight = inputHeight + padTop + padBottom

For Tiled Tensor:
effectiveHeight = inputHeight - (kernelHeight - 1)
                = inputHeight - overlap

This accounts for the memory needed to store
the kernel overlap region at tile boundaries.
```

---

## Examples

### Example 1: 3x3 Conv2D with Padding

**Configuration:**
- Input: `[1, 32, 32, 64]`
- Weight: `[128, 3, 3, 64]`
- Strides: `[1, 1]`
- Padding: `[1, 1, 1, 1]`
- L1 Memory: 32KB

**Constraints Applied:**

1. `outputHeight = (32 + 2 - 3 + 1) // 1 + 1 = 32`
2. `outputWidth = (32 + 2 - 3 + 1) // 1 + 1 = 32`
3. `inputChannels = 64` (full, not tiled)

**Resulting Tiles:**

| Tile | Output Shape | Output Offset | Input Shape | Input Offset | Padding |
|------|--------------|---------------|-------------|--------------|---------|
| 0 | `[1, 16, 32, 128]` | `[0, 0, 0, 0]` | `[1, 18, 32, 64]` | `[0, 0, 0, 0]` | `[1, 0, 1, 1]` |
| 1 | `[1, 16, 32, 128]` | `[0, 16, 0, 0]` | `[1, 18, 32, 64]` | `[0, 14, 0, 0]` | `[0, 1, 1, 1]` |

### Example 2: Strided Convolution

**Configuration:**
- Input: `[1, 16, 16, 32]`
- Weight: `[64, 3, 3, 32]`
- Strides: `[2, 2]`
- Padding: `[1, 1, 1, 1]`

**Output Size:**
- `outputHeight = (16 + 2 - 3 + 1) // 2 + 1 = 8`
- `outputWidth = (16 + 2 - 3 + 1) // 2 + 1 = 8`

**Stride Alignment:**
- Input tiles must have `effectiveHeight % 2 == 0`
- Input tiles must have `effectiveWidth % 2 == 0`

---

## Notes and Limitations

### Known Limitations

1. **Channel tiling not supported:** Input channels must be processed completely in each tile due to im2col algorithm requirements.

2. **Weight tiling not supported:** The complete kernel must be loaded for each tile computation.

3. **Minimum tile size:** The effective input tile size must be at least kernel size.

4. **Stride alignment:** Effective input dimensions must be divisible by stride.

### Platform-Specific Notes

- **Memory alignment:** Tensors should be 4-byte aligned for optimal DMA performance on PULP.
- **Im2Col buffer:** Additional transient buffer required for im2col transformation, sized as `n_cores * ch_im_in * kernel_h * kernel_w * dtype_width`.
- **Parallelization:** Output channel computation is parallelized across available cores.

### Performance Considerations

1. **Minimize overlap:** Larger output tiles reduce relative overhead from kernel overlap regions.
2. **Output channel alignment:** Keep output channels divisible by 8 for SIMD efficiency (when `ch_im_out >= 8`).
3. **Memory bandwidth:** Consider L2 to L1 bandwidth when selecting tile sizes.
4. **Transient buffer size:** Im2Col buffer scales with number of cores - balance parallelism vs memory usage.

### Differences: Conv2D vs RQConv2D

| Aspect | Conv2DTileConstraint | RQConv2DTileConstraint |
|--------|---------------------|------------------------|
| Data types | Float32 | Int8/Int32 |
| Requantization | No | Yes (mul, add buffers) |
| Bias handling | Optional bias buffer | Via add buffer |
| Output precision | Same as input | Configurable |

---

## References

- Source file: `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py`
- Templates: `Deeploy/Targets/PULPOpen/Templates/FloatConvTemplate.py`
- Tiler configuration: `Deeploy/Targets/PULPOpen/Tiler.py` (lines 49-53)
- ONNX Conv specification: https://onnx.ai/onnx/operators/onnx__Conv.html
