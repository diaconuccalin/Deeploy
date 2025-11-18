# Depthwise Convolution Tiling Constraints - Neureka

> **File:** `Deeploy/Targets/Neureka/TileConstraints/NeurekaDepthwiseConstraint.py`
> **Class:** `NeurekaDWConv2DTileConstraint`
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

This constraint class defines tiling rules for 3x3 Depthwise Convolution operations on the Neureka accelerator. In depthwise convolution, each input channel is convolved with its own set of filters independently, resulting in the same number of output channels as input channels. This is commonly used in efficient neural network architectures like MobileNets.

### Supported Data Types

| Input Type | Weight Type | Output Type |
|------------|-------------|-------------|
| `int8_t`   | `int8_t`    | `int8_t`    |
| `int8_t`   | `int8_t`    | `int32_t`   |

### Parent Class

- **Extends:** `TileConstraint`
- **Requantized variant:** `NeurekaRQSDWConv2DTileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[N, H, W, C]` | NHWC | Input activation tensor |
| `weight` | `[C, ...]` | Neureka format | Depthwise weight tensor |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[N, H_out, W_out, C]` | NHWC | Output activation tensor (same channels as input) |

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
| `ch_im_in` | `int` | Full input/output channels |
| `input_bits` | `int` | Input data bitwidth |
| `output_bits` | `int` | Output data bitwidth |

---

## Geometrical Constraints

Geometrical constraints define the mathematical relationships between tensor dimensions for valid depthwise convolution.

### Dimension Variables

| Variable | Tensor | Dimension Index | Description |
|----------|--------|-----------------|-------------|
| `inputBatchVar` | `data_in` | 0 | Batch size |
| `inputHeightVar` | `data_in` | 1 | Input height |
| `inputWidthVar` | `data_in` | 2 | Input width |
| `inputChannelVar` | `data_in` | 3 | Input channels |
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

**Description:** Batch dimension preserved through depthwise convolution.

---

#### Channel Dimension Constraint (Depthwise)

**Proposition:**
```
outputChannelVar == inputChannelVar
```

**Description:** **Key depthwise constraint** - output channels must equal input channels. Each input channel produces exactly one output channel.

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

#### Minimum Input Size Constraints

**Propositions:**
```
inputHeightVar >= 3
inputWidthVar >= 3
```

**Description:** Input tile must be at least 3x3 for valid 3x3 convolution.

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

**Description:** Standard convolution output formula with fixed 3x3 kernel.

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

**Description:** Standard convolution output formula with fixed 3x3 kernel.

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

#### Full Spatial Input Constraints (Performance Hints)

**Propositions:**
```
inputHeightVar == inputHeightVar.Max()
inputWidthVar == inputWidthVar.Max()
```

**Rationale:** Prefer full input height/width to minimize tiling overhead. Only relax if memory constraints require spatial tiling.

**Priority:** Performance hint (priority = 1)

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

    # Note: For depthwise, inCSize == CSize (channel correspondence)
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

2. **Channel correspondence:** Input and output channels must match. Tiling channels requires corresponding tiling of both input and output.

3. **Minimum input size:** Input tiles must be at least 3x3 spatially.

4. **No channel mixing:** Unlike dense convolution, depthwise does not mix information across channels.

### Platform-Specific Notes

- **Neureka 3x3 optimization:** Hardware optimized for 3x3 depthwise convolutions.
- **Stride computation:** `ioStridesFromDimensions()` computes byte strides.
- **Counter computation:** `Neureka2DDWConvTemplate.getCounters()` computes hardware iteration counters.

### Performance Considerations

1. **Prefer full input dimensions** - spatial tiling introduces overhead from kernel overlap
2. **Input at least 3x3** - hard requirement for valid convolution
3. **Channel tiling possible** - both input and output channels tile together
4. **WeightMemory_SRAM:** Use when available to eliminate weight transfer

### Tiling Strategy

```
Input Tensor [N, H, W, C]
+---------------------------+
|                           |
|   +-----------+           |
|   | Tile 0    |<-- Full H, W preferred
|   | C0..Ck    |<-- Can tile channels
|   +-----------+           |
|        |                  |
|   +-----------+           |
|   | Tile 1    |           |
|   | Ck+1..Cn  |<-- Channel correspondence
|   +-----------+           |
|                           |
+---------------------------+

Output Tensor [N, H_out, W_out, C]
- Same channel tiling as input (depthwise)
- Spatial dimensions computed from input + kernel
```

### Comparison with Dense Convolution

| Aspect | Dense Conv | Depthwise Conv |
|--------|------------|----------------|
| Channel mixing | Yes (all C_in -> each C_out) | No (each C -> each C) |
| Input channel tiling | Not allowed | Allowed (with output) |
| Output/Input channels | Independent | Must match |
| Computation | C_in * C_out per spatial | C per spatial |

### Requantization Support

The `NeurekaRQSDWConv2DTileConstraint` class extends this with requantization support:
- Adds `mul` and `add` tensors to geometrical constraints
- Includes requantization parameters in the tiling schedule

---

## References

- Source file: `Deeploy/Targets/Neureka/TileConstraints/NeurekaDepthwiseConstraint.py`
- Related templates: `Deeploy/Targets/Neureka/Templates/ConvTemplate.py`
- Helper functions: `Deeploy/Targets/Neureka/TileConstraints/RequantHelpers.py`
- Parent constraint: `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py`
