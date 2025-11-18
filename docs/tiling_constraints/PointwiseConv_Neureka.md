# Pointwise Convolution Tiling Constraints - Neureka

> **File:** `Deeploy/Targets/Neureka/TileConstraints/NeurekaPointwiseConstraint.py`
> **Class:** `NeurekaPWConv2DTileConstraint`
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

This constraint class defines tiling rules for 1x1 Pointwise Convolution operations on the Neureka accelerator. Pointwise convolutions are used for channel mixing without spatial filtering. The constraint handles the specific hardware subtiling requirements of the Neureka accelerator, which processes data in 6x6 spatial and 32-channel output blocks.

### Supported Data Types

| Input Type | Weight Type | Output Type |
|------------|-------------|-------------|
| `int8_t`   | `int8_t`    | `int8_t`    |
| `int8_t`   | `int8_t`    | `int32_t`   |

### Parent Class

- **Extends:** `TileConstraint`
- **Requantized variant:** `NeurekaRQSPWConv2DTileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[N, H, W, C_in]` | NHWC | Input activation tensor |
| `weight` | `[C_out, C_in_major, bandwidth]` | Neureka format | Weight tensor in Neureka layout |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[N, H, W, C_out]` | NHWC | Output activation tensor |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer |
| `data_out` | `str` | Name of output buffer |
| `weight` | `str` | Name of weight buffer |
| `strides` | `List[int]` | Convolution strides `[stride_h, stride_w]` |
| `pads` | `List[int]` | Padding `[pad_top, pad_left, pad_bottom, pad_right]` |
| `dim_im_out_x` | `int` | Full output height |
| `dim_im_out_y` | `int` | Full output width |
| `ch_im_out` | `int` | Full output channels |
| `ch_im_in` | `int` | Full input channels |
| `input_bits` | `int` | Input data bitwidth |
| `output_bits` | `int` | Output data bitwidth |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions for valid pointwise convolution.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `inputBatchVar` | `data_in` | 0 | Batch size |
| `inputHeightVar` | `data_in` | 1 | Input height |
| `inputWidthVar` | `data_in` | 2 | Input width |
| `weightOutChannelVar` | `weight` | 0 | Weight output channels |
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

**Description:** Batch dimension preserved through pointwise convolution.

---

#### Spatial Dimension Constraints

**Propositions:**
```
outputHeightVar == inputHeightVar
outputWidthVar == inputWidthVar
```

**Description:** For 1x1 pointwise convolution, spatial dimensions are preserved (same input and output height/width).

---

#### Weight-Output Channel Constraint

**Proposition:**
```
if weight is in WeightMemory_SRAM:
    weightOutChannelVar == weightOutChannelVar.Max()
else:
    weightOutChannelVar == outputChannelVar
```

**Description:** If weights are stored in dedicated SRAM, the full weight tensor is used. Otherwise, weights are tiled along with output channels.

---

#### Minimum Spatial Dimension Constraints

**Propositions:**
```
inputHeightVar >= 1
inputWidthVar >= 1
```

**Description:** Minimum tile size requirement - at least 1x1 spatial dimensions.

---

## Policy Constraints

Policy constraints define optimization hints and hardware-specific restrictions for the Neureka accelerator.

### Constraints

#### Full Input Channel Constraint

**Proposition:**
```
inputChannelVar == inputChannelVar.Max()
weightInChannelMajorVar == weightInChannelMajorVar.Max()
weightBandwidthVar == weightBandwidthVar.Max()
```

**Rationale:** Input channels cannot be tiled for convolution operations. The complete channel reduction must occur within a single tile to avoid partial results.

**Priority:** Hard constraint (must be satisfied)

---

#### Stride Alignment Constraints

**Propositions:**
```
(inputHeightVar % strides[0]) == 0
(inputWidthVar % strides[1]) == 0
```

**Rationale:** Input tile dimensions must be aligned with stride for correct output computation.

**Priority:** Hard constraint

---

#### Neureka Output Height Subtiling Constraint

**Proposition:**
```
if dim_im_out_x > 6:
    outputHeightVar % 6 == 0
else:
    outputHeightVar == outputHeightVar.Max()
```

**Rationale:** The Neureka accelerator processes output in 6x6 spatial blocks. When output height exceeds 6, keeping tiles divisible by 6 aligns with hardware subtiling and maximizes utilization.

**Priority:** Performance hint (priority = 3)

---

#### Neureka Output Width Subtiling Constraint

**Proposition:**
```
if dim_im_out_y > 6:
    outputWidthVar % 6 == 0
else:
    outputWidthVar == outputWidthVar.Max()
```

**Rationale:** Same as height - align with 6x6 hardware subtiling for width dimension.

**Priority:** Performance hint (priority = 2)

---

#### Neureka Output Channel Subtiling Constraint

**Proposition:**
```
if ch_im_out > 32:
    outputChannelVar % 32 == 0
else:
    outputChannelVar == outputChannelVar.Max()
```

**Rationale:** The Neureka accelerator processes 32 output channels at a time. Keeping output channels divisible by 32 ensures efficient hardware utilization.

**Priority:** Performance hint (priority = 1)

---

## Tiling Schedule

### Input Cube Computation

Input cubes are computed using `Conv2DTileConstraint.computeInputCube()` which handles padding:

**Algorithm:**

```python
for each output tile cube:
    (BatchOffset, HOffset, WOffset, COffset) = cube.offset
    (BatchSize, HSize, WSize, CSize) = cube.dims

    # Compute input cube with padding handling
    InCube, (padding_left, padding_right, padding_top, padding_bottom) = \
        Conv2DTileConstraint.computeInputCube(
            (weightH, weightW),  # 1x1 for pointwise
            pads, strides, weightC, cube, outputBuffer.shape
        )
```

### Load Schedule Structure

```python
inputLoadSchedule = [
    {"data_in": InCube},
    # weight is added if not in WeightMemory_SRAM
    ...
]

outputLoadSchedule = [
    {"data_out": outCube},
    ...
]
```

### Weight Memory Handling

If weights are in `WeightMemory_SRAM`:
- Weights are accessed via offset calculation (`weight_addr_offset`)
- No weight tiling needed

Otherwise:
- Weights are included in `inputLoadSchedule`
- Weight cube computed based on output channel offset

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

1. **No input channel tiling:** Full input channels required for convolution reduction.

2. **Hardware subtiling alignment:** For optimal performance, output tiles should align with Neureka's 6x6x32 subtiling.

3. **Weight memory modes:** Different handling based on whether weights are in dedicated SRAM or tiled memory.

### Platform-Specific Notes

- **Neureka subtiling:** The accelerator internally processes data in 6x6 spatial blocks and 32 output channel groups.
- **Stride computation:** `ioStridesFromDimensions()` computes byte strides based on bitwidth.
- **Counter computation:** `Neureka2DPWConvTemplate.getCounters()` computes hardware iteration counters.

### Performance Considerations

1. **Output height/width divisible by 6** for optimal Neureka spatial subtiling
2. **Output channels divisible by 32** for optimal Neureka channel subtiling
3. **Priority order:** Channel alignment (1) > Width alignment (2) > Height alignment (3)
4. **WeightMemory_SRAM:** When available, eliminates weight transfer overhead

### Requantization Support

The `NeurekaRQSPWConv2DTileConstraint` class extends this with requantization support:
- Adds `mul` and `add` tensors to geometrical constraints
- Includes requantization parameters in the tiling schedule

---

## References

- Source file: `Deeploy/Targets/Neureka/TileConstraints/NeurekaPointwiseConstraint.py`
- Related templates: `Deeploy/Targets/Neureka/Templates/ConvTemplate.py`
- Helper functions: `Deeploy/Targets/Neureka/TileConstraints/RequantHelpers.py`
