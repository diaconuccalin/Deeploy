# MaxPool Tiling Constraints - PULPOpen

> **File:** `Deeploy/Targets/PULPOpen/TileConstraints/MaxPoolTileConstraint.py`
> **Classes:** `MaxPoolHWTileConstraint`, `MaxPoolCTileConstraint`
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

This file defines tiling constraints for Max Pooling operations on the PULPOpen platform. Two different tiling strategies are provided:

1. **`MaxPoolHWTileConstraint`**: Tiles along height and width dimensions while keeping channels intact. This is the default spatial tiling approach.

2. **`MaxPoolCTileConstraint`**: Tiles along the channel dimension while keeping spatial dimensions intact. This is useful when the default memory level is L3 and padding margin calculations become problematic.

### Supported Data Types

| Input Type | Output Type |
|------------|-------------|
| `int8_t`   | `int8_t`    |
| `uint8_t`  | `uint8_t`   |

### Parent Class

- **Both classes inherit from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_in` | `[N, H, W, C]` | NHWC | Input activation tensor |

### Output Tensors

| Name | Shape | Layout | Description |
|------|-------|--------|-------------|
| `data_out` | `[N, H_out, W_out, C]` | NHWC | Output activation tensor |

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| `kernel_shape` | `List[int]` | Pooling kernel size `[K_H, K_W]` |
| `strides` | `List[int]` | Pooling strides `[stride_h, stride_w]` |
| `pads` | `List[int]` | Padding `[pad_top, pad_left, pad_bottom, pad_right]` |
| `ch_im_in` | `int` | Number of input channels |
| `dim_kernel_x` | `int` | Kernel height |
| `dim_kernel_y` | `int` | Kernel width |

---

## Geometrical Constraints

### MaxPoolHWTileConstraint

#### Dimension Variables

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

#### Constraints

##### Batch Dimension Constraint

**Proposition:**
```
outputBatchVar == inputBatchVar
```

**Description:** Batch dimension must be equal between input and output.

---

##### Channel Dimension Constraint

**Proposition:**
```
outputChannelVar == inputChannelVar
```

**Description:** Max pooling preserves channel count - no cross-channel operations.

---

##### Output Height Constraint

**Proposition:**
```
outputHeightVar == (effectiveHeight - (kernelShape[0] - 1) - 1) // strides[0] + 1
```

**Where:**
```
effectiveHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1]))
```

**Description:** Standard pooling output size formula. Padding is only added when processing the full input height.

---

##### Output Width Constraint

**Proposition:**
```
outputWidthVar == (effectiveWidth - (kernelShape[1] - 1) - 1) // strides[1] + 1
```

**Where:**
```
effectiveWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[2]))
```

**Description:** Same logic as height constraint, applied to width dimension.

---

### MaxPoolCTileConstraint

#### Constraints

##### Channel Dimension Match

**Proposition:**
```
outputDimVar[numDims - 1] == inputDimVar[numDims - 1]
```

**Description:** Only constrains that input and output channel dimensions are equal. This is the only geometrical constraint as spatial dimensions are kept fixed.

---

## Policy Constraints

### MaxPoolHWTileConstraint Policy Constraints

#### Full Channel Constraint

**Proposition:**
```
inputChannelVar == parseDict['ch_im_in']
```

**Rationale:** Channels must not be tiled for spatial tiling strategy. Max pooling operates independently per channel, but this constraint simplifies memory management.

**Priority:** Hard constraint

---

#### Minimum Input Height Constraint

**Proposition:**
```
inputHeightVar >= parseDict['dim_kernel_x']
```

**Rationale:** Input tile must be at least kernel height to apply pooling operation.

**Priority:** Hard constraint

---

#### Minimum Input Width Constraint

**Proposition:**
```
inputWidthVar >= parseDict['dim_kernel_y']
```

**Rationale:** Input tile must be at least kernel width to apply pooling operation.

**Priority:** Hard constraint

---

#### Stride Alignment Constraints

**Propositions:**
```
(inputHeightVar % strides[0]) == 0
(inputWidthVar % strides[1]) == 0
```

**Rationale:** Ensures input tiles are compatible with stride for correct output computation without partial results at tile boundaries.

**Priority:** Hard constraint

---

### MaxPoolCTileConstraint Policy Constraints

#### Fixed Spatial Dimensions

**Propositions:**
```
for idx in range(numDims):
    if idx != numDims - 1:  # All except channel dimension
        outputDimVar[idx] == outputBuffer.shape[idx]
        inputDimVar[idx] == inputBuffer.shape[idx]
```

**Rationale:** Keeps all dimensions except channels fixed at full size. This avoids issues with padding margin calculations when the default memory level is L3.

**Priority:** Hard constraint

---

## Tiling Schedule

### MaxPoolHWTileConstraint Schedule

Uses `Conv2DTileConstraint.computeInputCube` to calculate input tiles with proper overlap.

```python
for cube in outputCubes:
    (BatchOffset, HOffset, WOffset, COffset) = cube.offset
    (BatchSize, HSize, WSize, CSize) = cube.dims

    # Compute input cube with overlap and padding
    InCube, padding_tuple = Conv2DTileConstraint.computeInputCube(
        kernelShape=(kernel_h, kernel_w),
        pads=pads,
        strides=strides,
        inputCSize=CSize,
        outputCube=cube,
        outputDims=output_shape
    )
```

### MaxPoolCTileConstraint Schedule

Simpler computation since only channels are tiled:

```python
for cube in outputCubes:
    # Input cube has same offset but full spatial dimensions
    input_offset = list(cube.offset)
    input_dims = list(input_shape)
    input_dims[-1] = cube.dims[-1]  # Only channel varies
    InCube = HyperRectangle(tuple(input_offset), tuple(input_dims))
```

### Load Schedule Structure

```python
# MaxPoolHWTileConstraint
inputLoadSchedule = [
    {"data_in": InCube},
    ...
]

outputLoadSchedule = [
    {"data_out": OutCube},
    ...
]
```

---

## Replacements

### MaxPoolHWTileConstraint Replacements

| Variable | Type | Description |
|----------|------|-------------|
| `dim_im_in_x` | `uint16_t` | Tiled input height |
| `dim_im_in_y` | `uint16_t` | Tiled input width |
| `dim_im_out_x` | `uint16_t` | Tiled output height |
| `dim_im_out_y` | `uint16_t` | Tiled output width |
| `ch_im_in` | `uint16_t` | Input channels (always full) |
| `padding_y_top` | `uint8_t` | Top padding for tile |
| `padding_y_bottom` | `uint8_t` | Bottom padding for tile |
| `padding_x_left` | `uint8_t` | Left padding for tile |
| `padding_x_right` | `uint8_t` | Right padding for tile |

### MaxPoolCTileConstraint Replacements

| Variable | Type | Description |
|----------|------|-------------|
| `ch_im_in` | `uint16_t` | Tiled channel count |

---

## Visual Explanation

### MaxPoolHWTileConstraint - Spatial Tiling

```
Input Tensor [N, H, W, C]
+--------------------------------+
|  +---------+                   |
|  | Tile 0  | <- All channels   |
|  +---------+                   |
|       |                        |
|  +---------+                   |
|  | Tile 1  | <- Overlap for    |
|  +---------+    kernel window  |
|                                |
+--------------------------------+

Output Tensor [N, H_out, W_out, C]
+--------------------------------+
|  +-----+                       |
|  | T0  |                       |
|  +-----+                       |
|  +-----+                       |
|  | T1  | <- No overlap         |
|  +-----+                       |
+--------------------------------+
```

### MaxPoolCTileConstraint - Channel Tiling

```
Input Tensor [N, H, W, C]
+--------------------------------+
|                                |
|  Full spatial dimensions       |
|  [N, H, W, C_tile]             |
|                                |
+--------------------------------+
         |
         v  Tile on C only
+----------------+
|  Channels 0-31 |
+----------------+
|  Channels 32-63|
+----------------+
```

### Pooling Operation

```
Input Window:          Output:
+---+---+---+
| 1 | 3 | 2 |
+---+---+---+
| 5 | 9 | 4 |    ->    [ 9 ]  (max value)
+---+---+---+
| 6 | 7 | 8 |
+---+---+---+
```

### Input Tile Overlap

```
Output Tiles:           Input Tiles (with overlap):
+-------+              +------------+
|   0   |              |  Tile 0    |
+-------+              +--+------+--+
|   1   |                 | Ovlp |
+-------+              +--+------+--+
                       |  Tile 1    |
                       +------------+

Overlap = kernel_size - 1
```

---

## Examples

### Example 1: 2x2 Max Pooling with Stride 2

**Input Parameters:**
- Input shape: `[1, 32, 32, 64]`
- Kernel shape: `[2, 2]`
- Strides: `[2, 2]`
- Padding: `[0, 0, 0, 0]`
- Output shape: `[1, 16, 16, 64]`
- L1 Memory: 16KB

**Tiling Result (MaxPoolHWTileConstraint):**

- Output tile 0: Shape `[1, 8, 16, 64]`, Offset `[0, 0, 0, 0]`
- Output tile 1: Shape `[1, 8, 16, 64]`, Offset `[0, 8, 0, 0]`

**Input Tiles:**
- Input tile 0: Shape `[1, 16, 32, 64]`, Offset `[0, 0, 0, 0]`
- Input tile 1: Shape `[1, 16, 32, 64]`, Offset `[0, 16, 0, 0]`

**Note:** With stride 2 and kernel 2, there's no overlap between input tiles.

---

### Example 2: 3x3 Max Pooling with Stride 1

**Input Parameters:**
- Input shape: `[1, 32, 32, 128]`
- Kernel shape: `[3, 3]`
- Strides: `[1, 1]`
- Padding: `[1, 1, 1, 1]`
- Output shape: `[1, 32, 32, 128]`

**Input Tiles (with overlap):**
- Input tile 0: Shape `[1, 18, 32, 128]`, Offset `[0, 0, 0, 0]`, Padding `[1, 1, 0, 1]`
- Input tile 1: Shape `[1, 18, 32, 128]`, Offset `[0, 15, 0, 0]`, Padding `[0, 1, 1, 1]`

**Overlap calculation:** kernel_size - 1 = 2 rows

---

### Example 3: Channel Tiling for L3 Memory

**Scenario:** When input/output tensors are in L3 memory, use `MaxPoolCTileConstraint`.

**Input Parameters:**
- Input shape: `[1, 64, 64, 256]`
- Memory level: L3

**Tiling Result:**

Spatial dimensions stay fixed, only channels are tiled:
- Tile 0: `[1, 64, 64, 64]`, Channel offset 0
- Tile 1: `[1, 64, 64, 64]`, Channel offset 64
- Tile 2: `[1, 64, 64, 64]`, Channel offset 128
- Tile 3: `[1, 64, 64, 64]`, Channel offset 192

---

## Notes and Limitations

### Known Limitations

1. **MaxPoolHWTileConstraint:**
   - Cannot tile channels - must process all channels together
   - Requires stride alignment of input tile dimensions

2. **MaxPoolCTileConstraint:**
   - Cannot tile spatial dimensions - must process full H and W
   - Useful primarily when L3 memory is involved

3. **No Global Pooling support:** These constraints are for local max pooling with fixed kernel sizes. Global pooling requires different handling.

### Platform-Specific Notes

- **Memory alignment:** Tensors should be aligned for optimal DMA performance
- **Padding handling:** The Im2Col-style kernel handles zero-padding internally, requiring proper padding values per tile

### Performance Considerations

1. **Choose appropriate strategy:**
   - Use `MaxPoolHWTileConstraint` when tensors fit in L1/L2
   - Use `MaxPoolCTileConstraint` when dealing with L3 memory or very large spatial dimensions

2. **Minimize overlap:** Choose larger output tiles to reduce the relative overhead of input overlap:
   - Overlap cost = (kernel_size - 1) / tile_size
   - Larger tiles = lower relative overhead

3. **Stride considerations:**
   - stride >= kernel_size: No overlap needed
   - stride < kernel_size: Overlap = kernel_size - stride

4. **Memory bandwidth:** Max pooling is memory-bound (simple max operation), so optimize for data movement.

### When to Use Each Constraint

| Scenario | Recommended Constraint |
|----------|------------------------|
| Standard pooling, L1/L2 memory | `MaxPoolHWTileConstraint` |
| L3 memory default level | `MaxPoolCTileConstraint` |
| Small spatial, large channels | `MaxPoolCTileConstraint` |
| Large spatial, small channels | `MaxPoolHWTileConstraint` |

---

## References

- Source file: `Deeploy/Targets/PULPOpen/TileConstraints/MaxPoolTileConstraint.py`
- Input cube computation: `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py`
- Related templates: `Deeploy/Targets/PULPOpen/Templates/`
- Tiler configuration: `Deeploy/Targets/PULPOpen/Tiler.py`
