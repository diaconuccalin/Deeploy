# Dense Convolution Tiling Constraints - Neureka

> **File:** `Deeploy/Targets/Neureka/TileConstraints/NeurekaDenseConstraint.py`
> **Class:** `NeurekaDenseConv2DTileConstraint`
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

This constraint class defines tiling rules for 3x3 Dense (standard) Convolution operations on the Neureka accelerator. Dense convolutions perform spatial filtering with learned kernels. The Neureka accelerator is optimized for 3x3 kernel operations and uses specific subtiling patterns for efficient processing.

### Supported Data Types

| Input Type | Weight Type | Output Type |
|------------|-------------|-------------|
| `int8_t`   | `int8_t`    | `int8_t`    |
| `int8_t`   | `int8_t`    | `int32_t`   |

### Parent Class

- **Extends:** `TileConstraint`
- **Requantized variant:** `NeurekaRQSDenseConv2DTileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[N, H, W, C_in]` | NHWC | Input activation tensor |
| `weight` | `[C_out, C_in_major, bits, bandwidth]` | Neureka format | Weight tensor in Neureka layout |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[N, H_out, W_out, C_out]` | NHWC | Output activation tensor |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer |
| `data_out` | `str` | Name of output buffer |
| `weight` | `str` | Name of weight buffer |
| `strides` | `List[int]` | Convolution strides `[stride_h, stride_w]` |
| `pads` | `List[int]` | Padding `[pad_top, pad_left, pad_bottom, pad_right]` |
| `dilations` | `List[int]` | Dilation factors `[dilation_h, dilation_w]` |
| `dim_kernel_x` | `int` | Kernel height (3) |
| `dim_kernel_y` | `int` | Kernel width (3) |
| `ch_im_in` | `int` | Full input channels |
| `input_bits` | `int` | Input data bitwidth |
| `output_bits` | `int` | Output data bitwidth |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions for valid dense convolution.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `inputBatchVar` | `data_in` | 0 | Batch size |
| `inputHeightVar` | `data_in` | 1 | Input height |
| `inputWidthVar` | `data_in` | 2 | Input width |
| `inputChannelVar` | `data_in` | 3 | Input channels |
| `weightOutChannelVar` | `weight` | 0 | Weight output channels |
| `weightInChannelMajorVar` | `weight` | 1 | Weight input channel major |
| `weightBitsVar` | `weight` | 2 | Weight bits dimension |
| `weightBandwidthVar` | `weight` | 3 | Weight bandwidth |
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

**Description:** Batch dimension preserved through convolution.

---

#### Weight-Output Channel Constraint

**Proposition:**
```
if weight is in WeightMemory_SRAM:
    weightOutChannelVar == weightOutChannelVar.Max()
else:
    weightOutChannelVar == outputChannelVar
```

**Description:** If weights are in dedicated SRAM, full weights are used. Otherwise, weights are tiled with output channels.

---

#### Output Height Constraint

**Proposition:**
```
outputHeightVar == (effectiveHeight - (3 - 1) - 1) // strides[0] + 1
```

**Where:**
```
effectiveHeight = inputHeightVar + (
    (padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[1])
)
```

**Description:**
- For full tensor tiles: padding is added to effective height
- For partial tiles: no padding adjustment needed
- Uses standard convolution output formula with fixed 3x3 kernel

---

#### Output Width Constraint

**Proposition:**
```
outputWidthVar == (effectiveWidth - (3 - 1) - 1) // strides[1] + 1
```

**Where:**
```
effectiveWidth = inputWidthVar + (
    (padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[2])
)
```

**Description:** Same formula as height, applied to width dimension.

---

## Policy Constraints

Policy constraints define optimization hints and hardware-specific restrictions for the Neureka accelerator.

### Constraints

#### Stride Alignment Constraints

**Propositions:**
```
(inputHeightVar % strides[0]) == 0
(inputWidthVar % strides[1]) == 0
```

**Rationale:** Input tile dimensions must align with stride for correct output computation.

**Priority:** Hard constraint

---

#### Full Input Channel Constraint

**Proposition:**
```
inputChannelVar == inputChannelVar.Max()
```

**Rationale:** Input channels cannot be tiled. Complete channel reduction required within each tile.

**Priority:** Hard constraint

---

#### Full Spatial Input Constraints (Performance Hints)

**Propositions:**
```
inputHeightVar == inputHeightVar.Max()
inputWidthVar == inputWidthVar.Max()
```

**Rationale:** Prefer full input height/width to minimize tiling overhead and maximize Neureka utilization. Only relax if memory constraints require tiling.

**Priority:** Performance hint (priority = 1)

---

#### Minimum Input Size Constraints

**Propositions:**
```
inputHeightVar >= dim_kernel_x  (kernel height = 3)
inputWidthVar >= dim_kernel_y   (kernel width = 3)
```

**Rationale:** Input tile must be at least kernel size to produce valid output.

**Priority:** Hard constraint

---

## Tiling Schedule

### Input Cube Computation

Input cubes are computed using `Conv2DTileConstraint.computeInputCube()`:

**Algorithm:**

```python
for each output tile cube:
    (BatchOffset, HOffset, WOffset, COffset) = cube.offset
    (BatchSize, HSize, WSize, CSize) = cube.dims

    # Compute input cube with padding handling
    InCube, (padding_left, padding_right, padding_top, padding_bottom) = \
        Conv2DTileConstraint.computeInputCube(
            (weightH, weightW),  # 3x3
            pads, strides, weightC, cube, outputBuffer.shape
        )

    inBSize, inHSize, inWSize, inCSize = InCube.dims
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": InCube},
    # weight added if not in WeightMemory_SRAM
    ...
]

outputLoadSchedule = [
    {"data_out": outCube},
    ...
]
```

### Weight Memory Handling

If weights are in `WeightMemory_SRAM`:
- Use `weight_addr_offset` for offset calculation
- No weight transfer needed

Otherwise:
- Include weights in `inputLoadSchedule`
- Weight cube: `HyperRectangle((COffset, 0, 0), (CSize, weightShape[-2], weightShape[-1]))`

---

## Replacements

Variables that are replaced with tile-specific values during code generation.

| Variable | Type | Description |
|----------|------|-------------|
| `padding_y_top` | `uint8_t*` | Top padding for tile |
| `padding_y_bottom` | `uint8_t*` | Bottom padding for tile |
| `padding_x_left` | `uint8_t*` | Left padding for tile |
| `padding_x_right` | `uint8_t*` | Right padding for tile |
| `dim_im_in_x_stride` | `uint32_t*` | Input X stride in bytes |
| `dim_im_in_y_stride` | `uint32_t*` | Input Y stride in bytes |
| `dim_im_out_x_stride` | `uint32_t*` | Output X stride in bytes |
| `dim_im_out_y_stride` | `uint32_t*` | Output Y stride in bytes |
| `input_addr_offset` | `uint32_t*` | Input address offset for padding |
| `nKo` | `uint16_t*` | Number of output channel iterations |
| `nKi` | `uint16_t*` | Number of input channel iterations |
| `nHo` | `uint16_t*` | Number of output height iterations |
| `nWo` | `uint16_t*` | Number of output width iterations |
| `bKo` | `uint16_t*` | Output channel block size |
| `bKi` | `uint16_t*` | Input channel block size |
| `bHo` | `uint16_t*` | Output height block size |
| `bWo` | `uint16_t*` | Output width block size |
| `bHi` | `uint16_t*` | Input height block size |
| `bWi` | `uint16_t*` | Input width block size |
| `weight_addr_offset` | `uint32_t*` | Weight address offset (if WeightMemory_SRAM) |

---

## Notes and Limitations

### Known Limitations

1. **Fixed 3x3 kernel:** The Neureka accelerator is optimized for 3x3 kernels. This constraint assumes kernel size of 3x3.

2. **No input channel tiling:** Full input channels required for complete reduction.

3. **Minimum input size:** Input tiles must be at least 3x3 to produce valid output.

4. **Spatial tiling overhead:** When input height/width must be tiled, kernel overlap creates data redundancy.

### Platform-Specific Notes

- **Neureka 3x3 optimization:** The accelerator has dedicated hardware for 3x3 convolutions.
- **Stride computation:** `ioStridesFromDimensions()` computes byte strides based on bitwidth.
- **Counter computation:** `Neureka2DDenseConvTemplate.getCounters()` computes hardware iteration counters.

### Performance Considerations

1. **Prefer full input dimensions** - tiling spatial dimensions introduces overhead
2. **Input at least 3x3** - hard requirement for valid convolution
3. **Channel tiling:** Only output channels can be tiled
4. **WeightMemory_SRAM:** Use when available to eliminate weight transfer

### Tiling Strategy

```
Input Tensor [N, H, W, C_in]
+---------------------------+
|                           |
|   +-----------+           |
|   | Tile 0    |<-- Full H, W preferred
|   | Full C_in |
|   +-----------+           |
|        |                  |
|   +-----------+           |
|   | Tile 1    |<-- Tile output channels
|   | Full C_in |           |
|   +-----------+           |
|                           |
+---------------------------+

- Input channels: Full (no tiling)
- Output channels: Tileable
- Spatial: Prefer full, tile if memory constrained
```

### Requantization Support

The `NeurekaRQSDenseConv2DTileConstraint` class extends this with requantization support:
- Adds `mul` and `add` tensors to geometrical constraints
- Includes requantization parameters in the tiling schedule

---

## References

- Source file: `Deeploy/Targets/Neureka/TileConstraints/NeurekaDenseConstraint.py`
- Related templates: `Deeploy/Targets/Neureka/Templates/ConvTemplate.py`
- Helper functions: `Deeploy/Targets/Neureka/TileConstraints/RequantHelpers.py`
- Parent constraint: `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py`
