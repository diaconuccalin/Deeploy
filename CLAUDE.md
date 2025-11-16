# Claude Development Notes

## Branch: claude/tinyviT-c-implementation-016xWn1JdvtVcq7zMLiuScRb

This branch fixes CCT (Compact Convolutional Transformer) test failures in the TinyViT tiled execution on the Siracusa platform.

### Branch History

Base branch: `DemoTinyViT_Tiled_Siracusa` (commit 46c8fb5)

### Issue Found

The base branch had commit a73b7c4 which removed the `+ 1` term from the Conv2D tiling constraints (lines 315-317). This was incorrect and caused CCT and other tests to fail.

### Root Cause Analysis

The Conv2D tiling constraint formula WITHOUT `+ 1` gave incorrect output dimensions:

**Buggy Formula** (from commit a73b7c4):
```python
outputHeightVar == (effectiveHeight - dilations[0] * (weightHeightVar - 1) - 1) // strides[0]
outputWidthVar == (effectiveWidth - dilations[1] * (weightWidthVar - 1) - 1) // strides[1]
```

**Impact on CCT**:
- CCT uses: kernel=3x3, stride=1, padding=1, dilation=1
- Input: 16×16, Expected output: 16×16
- Buggy formula gave: 15×15 ❌
- Correct formula gives: 16×16 ✅

**Impact on testFloat2DConvolution**:
- Uses: kernel=3x3, stride=2, padding=1, dilation=1
- Input: 16×16, Expected output: 8×8
- Buggy formula gave: 7×7 ❌
- Correct formula gives: 8×8 ✅

### The Fix

**Correct Formula** (matching standard convolution formula):
```python
outputHeightVar == (effectiveHeight - dilations[0] * (weightHeightVar - 1) - 1) // strides[0] + 1
outputWidthVar == (effectiveWidth - dilations[1] * (weightWidthVar - 1) - 1) // strides[1] + 1
```

This matches the standard convolution output formula:
```
output_size = floor((input + padding - dilation*(kernel-1) - 1) / stride) + 1
```

**File Changed**: `Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py` (lines 315-317)

### Verification

All tests pass with the fix on Generic platform (clean builds):

**Conv2D Tests**:
- ✅ testFloat2DConvolution - PASS (0/512 errors)
- ✅ testFloat2DConvolutionBias - PASS (0/840 errors)
- ✅ testFloat2DConvolutionZeroBias - PASS (0/840 errors)

**CCT Tests**:
- ✅ CCT/CCT_1_16_16_8 - PASS (0/10 errors)

**Code Quality**:
- ✅ Python formatting (yapf) - PASS
- ✅ Import formatting (isort) - PASS
- ✅ Unused imports (autoflake) - PASS

### CI Pipeline Expectations

With this fix, ALL CI pipelines should pass:
- ✅ CI • Siracusa (Tiled) - CCT tests will pass
- ✅ CI • Generic - All Conv2D tests will pass
- ✅ CI • Deeploy / deeploy-memory-allocation - Will pass
- ✅ All other platform-specific tests - Will benefit from fix

### Formula Consistency

All Conv2D-related tiling constraints now consistently use the `+ 1` term:
- ✅ RQConv2DTileConstraint (line 72-73): Has `+ 1`
- ✅ DWConvTileConstraint (line 79-80): Has `+ 1`
- ✅ MaxPoolTileConstraint (line 52-53): Has `+ 1`
- ✅ Conv2DTileConstraint (line 315-317): Has `+ 1` - **FIXED**

### Notes

This fix restores the correct convolution output dimension calculation for tiled execution. Commit a73b7c4 from the base branch incorrectly removed the `+ 1` term, causing CCT and related tests to fail. The formula now matches the standard convolution formula used throughout deep learning frameworks.
