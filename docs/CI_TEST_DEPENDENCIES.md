# CI Test Dependency Graphs

This document provides comprehensive dependency mappings between operators, tests, platforms, and components to help identify which tests to run when making changes to the Deeploy codebase.

---

## Table of Contents

1. [Operator to Test Mapping](#operator-to-test-mapping)
2. [Platform to Test Mapping](#platform-to-test-mapping)
3. [Component to Test Mapping](#component-to-test-mapping)
4. [Test Category Index](#test-category-index)
5. [Change Impact Analysis Guide](#change-impact-analysis-guide)

---

## Operator to Test Mapping

When modifying an operator implementation, run all tests that use that operator.

### Core Operators

#### Add
**Tests:** Adder, MultIO, Attention, TestAdderLarge, TestRQAdd, largeFloatAdd, testFloatAdder, testFloatReshapeWithSkipConnection, testMatMulAdd, testTypeInferenceDifferentTypes, CCT/*, ICCT/*, microLlama/*, miniMobileNetv2, MLPerf/ImageClassification, Quant, QuantizedLinear, Transformer, WaveFormer, testFloatDemoTinyViT, testRQGEMMwBatch

**Platforms:** Generic, CortexM, MemPool, Snitch, Siracusa, Siracusa-Tiled, Snitch-Tiled, Neureka

---

#### Conv (2D Convolution)
**Tests:** test1DConvolution, test2DConvolution, test1DDWConvolution, test2DDWConvolution, test2DRequantizedConv, test2DRequantizedStriddedPaddedConv, testFloat2DConvolution, testFloat2DConvolutionBias, testFloat2DConvolutionZeroBias, testFloat2DDWConvolution, testFloat2DDWConvolutionBias, testFloat2DDWConvolutionZeroBias, testRQConv, testRequantizedDWConv, testPointwise, testPointwiseConvBNReLU, testPointwiseUnsignedWeights, simpleCNN, simpleRegression, miniMobileNet, miniMobileNetv2, MobileNetv2, CCT/*, ICCT/*, MLPerf/*, WaveFormer, Autoencoder1D, testFloatDemoTinyViT

**Platforms:** Generic, MemPool, Siracusa, Siracusa-Tiled, Neureka

**TileConstraints:** Conv2DTileConstraint, DWConv2DTileConstraint, RQConv2DTileConstraint

---

#### MatMul
**Tests:** testMatMul, testMatMulAdd, testMatMulBatch, testFloatMatmul, testRQMatMul, Attention, CCT/*, ICCT/*, microLlama/*, Transformer, WaveFormer, EEGFormer, QuantizedLinear, testFloatDemoTinyViT, testRQGEMMwBatch

**Platforms:** Generic, CortexM, MemPool, Snitch, Siracusa, Siracusa-Tiled, Snitch-Tiled

**TileConstraints:** MatMulTileConstraint

---

#### Gemm
**Tests:** testGEMM, testFloatGEMM, testFloatGEMMnobias, testFloatGEMMtransB, testRQGEMM, testRQGEMMTransB, testRequantizedLinear, simpleRegression, miniMobileNet, miniMobileNetv2, CCT/*, ICCT/*, MLPerf/*, WaveFormer, Autoencoder1D, testFloatReshapeWithSkipConnection, testFloatDemoTinyViT

**Platforms:** Generic, MemPool, Snitch, Siracusa, Siracusa-Tiled, Snitch-Tiled, Neureka

**TileConstraints:** GEMMTileConstraint, FloatGEMMTileConstraint

---

#### MaxPool
**Tests:** testMaxPool, testFloatMaxPool, simpleCNN, simpleRegression, CCT/*, ICCT/*, MLPerf/*, Autoencoder1D

**Platforms:** Generic, CortexM, MemPool, Siracusa, Siracusa-Tiled

**TileConstraints:** MaxPoolTileConstraint, MaxPoolHWTileConstraint, MaxPoolCTileConstraint

---

#### Softmax
**Tests:** testFloatSoftmax, CCT/*, testFloatDemoTinyViT

**Platforms:** Generic, Snitch, Siracusa, Siracusa-Tiled, Snitch-Tiled

---

#### iSoftmax (Integer Softmax)
**Tests:** iSoftmax, TestiSoftmaxLarge, Attention, Transformer, microLlama/*

**Platforms:** Generic, Snitch, Siracusa, Siracusa-Tiled, Snitch-Tiled

**TileConstraints:** iSoftmaxTileConstraint

---

#### LayerNormalization
**Tests:** testFloatLayerNorm, CCT/*, testFloatDemoTinyViT

**Platforms:** Generic, Siracusa, Siracusa-Tiled

**TileConstraints:** LayernormTileConstraint

---

#### Relu
**Tests:** testFloatRelu, CCT/*, Autoencoder1D

**Platforms:** Generic, Siracusa, Siracusa-Tiled

---

#### Gelu
**Tests:** testFloatGelu, CCT/*, testFloatDemoTinyViT

**Platforms:** Generic, Siracusa

---

#### Transpose
**Tests:** testFloatTranspose, Attention, CCT/*, ICCT/*, microLlama/*, Transformer, WaveFormer, EEGFormer, QuantizedLinear, testFloatDemoTinyViT

**Platforms:** Generic, Siracusa, Siracusa-Tiled

**TileConstraints:** TransposeTileConstraint

---

#### Reshape
**Tests:** testFloatReshape, testFloatReshapeWithSkipConnection, Attention, CCT/*, ICCT/*, microLlama/*, Transformer, WaveFormer, EEGFormer, MLPerf/*, Autoencoder1D, testFloatDemoTinyViT

**Platforms:** Generic, Siracusa

---

#### Concat
**Tests:** testConcat, testBacktracking, CCT/CCT_2_32_32_128_Opset20, microLlama/*, testFloatDemoTinyViT

**Platforms:** Siracusa, Siracusa-Tiled

**TileConstraints:** ConcatTileConstraint

---

#### ReduceMean
**Tests:** testReduceMean, testFloatDemoTinyViT

**Platforms:** Generic, CortexM, Siracusa, Siracusa-Tiled

**TileConstraints:** ReduceMeanTileConstraint

---

#### ReduceSum
**Tests:** testReduceSum, testFloatReduceSum

**Platforms:** Generic, CortexM, MemPool, Siracusa

---

#### Pad
**Tests:** test1DPad, test2DPad, testFloat2DPadding

**Platforms:** Generic, CortexM, MemPool, Siracusa

---

#### Slice
**Tests:** testSlice, CCT/CCT_2_32_32_128_Opset20, testFloatDemoTinyViT

**Platforms:** Generic, CortexM, MemPool, Siracusa

---

#### RequantShift
**Tests:** test2DRequantizedConv, test2DRequantizedStriddedPaddedConv, testRequantizedDWConv, testRQConv, testRQMatMul, testRQGEMM, testRQGEMMTransB, testRQGEMMwBatch, TestRQAdd, RQHardswish, testRequantizedLinear, testPointwiseConvBNReLU, testPointwiseUnsignedWeights, trueIntegerDivSandwich, Attention, ICCT/*, microLlama/*, miniMobileNet, miniMobileNetv2, MobileNetv2, MLPerf/*, simpleRegression, simpleCNN, Transformer, WaveFormer

**Platforms:** Generic, MemPool, Snitch, Siracusa, Siracusa-Tiled, Snitch-Tiled, Neureka

---

#### iHardswish
**Tests:** Hardswish, RQHardswish, microLlama/*

**Platforms:** Siracusa, Siracusa-Tiled

---

#### iRMSNorm
**Tests:** testRMSNorm, microLlama/*

**Platforms:** Siracusa, Siracusa-Tiled

---

#### Mul
**Tests:** testFloatMul, CCT/*, microLlama/*, WaveFormer, testFloatDemoTinyViT, Dequant

**Platforms:** Generic, Siracusa, Siracusa-Tiled

---

#### Div
**Tests:** testFloatDiv, Quant, QuantizedLinear

**Platforms:** Generic, Siracusa

---

#### TrueIntegerDiv
**Tests:** trueIntegerDivSandwich, microLlama/*

**Platforms:** Siracusa, Siracusa-Tiled

---

### Special/Platform-Specific Operators

#### ITAMax / ITAPartialMax
**Tests:** ICCT, ICCT_8, ICCT_ITA, ICCT_ITA_8

**Platforms:** Generic, MemPool

---

#### iLayerNorm
**Tests:** ICCT/*, ICCT_ITA/*

**Platforms:** Generic, MemPool

---

#### iGELU
**Tests:** ICCT/*, WaveFormer

**Platforms:** Generic, MemPool

---

#### IntegerMean
**Tests:** ICCT/*, WaveFormer

**Platforms:** Generic, MemPool

---

#### iNoNorm
**Tests:** TestiNoNorm

**Platforms:** Snitch, Snitch-Tiled

---

---

## Platform to Test Mapping

### Generic (Host CPU)

**Workflow:** ci-platform-generic.yml
**Runner:** testRunner_generic.py
**Test Count:** 52

**Kernel Tests:**
- Adder, MultIO
- test1DConvolution, test2DConvolution, test1DDWConvolution, test2DDWConvolution
- test1DPad, test2DPad
- testGEMM, testMatMul, testMatMulAdd, testMaxPool
- testRQConv, testRQMatMul, testReduceSum, testReduceMean, testSlice
- testRequantizedDWConv, test2DRequantizedConv
- iSoftmax
- testFloatAdder, testFloatGEMM
- testFloat2DConvolution, testFloat2DConvolutionBias, testFloat2DConvolutionZeroBias
- testFloatLayerNorm, testFloatDiv
- testFloat2DDWConvolution, testFloat2DDWConvolutionBias, testFloat2DDWConvolutionZeroBias
- testFloatRelu, testFloatMaxPool, testFloatMatmul
- testFloatReshapeWithSkipConnection, testFloatSoftmax, testFloatTranspose, testFloatMul
- Quant, Dequant, QuantizedLinear

**Model Tests:**
- simpleRegression, WaveFormer, simpleCNN
- ICCT, ICCT_ITA, ICCT_8, ICCT_ITA_8
- miniMobileNet, miniMobileNetv2
- CCT/CCT_1_16_16_8, CCT/CCT_2_32_32_128_Opset20
- testFloatDemoTinyViT, Autoencoder1D

---

### CortexM (ARM Cortex-M4)

**Workflow:** ci-platform-cortexm.yml
**Runner:** testRunner_cortexm.py
**Simulator:** QEMU
**Test Count:** 13

**Kernel Tests:**
- Adder, MultIO
- test1DPad, test2DPad
- testMatMul, testMatMulAdd, testMaxPool
- testRQConv, testReduceSum, testReduceMean, testSlice

**Model Tests:**
- simpleRegression, WaveFormer

---

### MemPool (Multi-core RISC-V)

**Workflow:** ci-platform-mempool.yml
**Runner:** testRunner_mempool.py
**Simulator:** Banshee
**Test Count:** 27

**Kernel Tests:**
- Adder, MultIO
- test1DConvolution, test2DConvolution, test1DDWConvolution, test2DDWConvolution
- test1DPad, test2DPad
- testGEMM, testMatMul, testMatMulAdd, testMaxPool
- testRQConv, testRQGEMM, testRQMatMul
- testReduceSum, testReduceMean, testSlice
- testRequantizedDWConv, test2DRequantizedConv

**Model Tests:**
- simpleRegression, simpleCNN
- ICCT, ICCT_ITA, ICCT_8
- miniMobileNet, miniMobileNetv2

---

### Snitch (RISC-V Cluster)

**Workflow:** ci-platform-snitch.yml
**Runner:** testRunner_snitch.py
**Simulator:** GVSoC
**Cores:** 9
**Test Count:** 10

**Kernel Tests:**
- Adder, iSoftmax
- TestiNoNorm, TestAdderLarge, TestiSoftmaxLarge
- testMatMul, testRQGEMM, TestRQAdd, testRQGEMMTransB
- testFloatSoftmax

---

### Snitch-Tiled

**Workflow:** ci-platform-snitch-tiled.yml
**Runner:** testRunner_tiled_snitch.py
**Simulator:** GVSoC
**Cores:** 9
**Test Count:** 8

**Tests with L1 configurations:**
- TestiNoNorm: [5000, 10000]
- TestAdderLarge: [5000, 10000]
- TestiSoftmaxLarge: [5000, 10000]
- testRQGEMM: [2000, 5000]
- testFloatSoftmax: [2000, 5000, 10000]
- TestRQAdd: [5000, 10000]
- testFloatGEMM: [2000, 5000, 10000]
- testFloatGEMMtransB: [2000, 5000, 10000]

---

### Siracusa (PULP Cluster)

**Workflow:** ci-platform-siracusa.yml
**Runner:** testRunner_siracusa.py
**Simulator:** GVSoC
**Cores:** 8
**Test Count:** 49

**Kernel Tests:**
- Adder, MultIO
- test1DPad, test2DPad
- testMatMul, testMatMulAdd
- testRequantizedDWConv, test2DRequantizedConv
- iSoftmax, testConcat, testRMSNorm, trueIntegerDivSandwich
- Hardswish, RQHardswish, testBacktracking
- testFloatAdder, testFloatGEMM
- testFloat2DConvolution, testFloat2DConvolutionBias, testFloat2DConvolutionZeroBias
- testFloat2DDWConvolution, testFloat2DDWConvolutionBias, testFloat2DDWConvolutionZeroBias
- testFloatLayerNorm, testFloatRelu, testFloatMaxPool, testFloatMatmul
- testFloatSoftmax, testFloatTranspose, testFloatMul
- Quant, Dequant, testFloatReduceSum, testFloatReshapeWithSkipConnection
- testFloatSoftmaxGrad, testFloatSoftmaxCrossEntropy, testFloatSoftmaxCrossEntropyGrad
- QuantizedLinear

**Model Tests:**
- simpleRegression, miniMobileNet, miniMobileNetv2
- Attention
- MLPerf/KeywordSpotting, MLPerf/ImageClassification, MLPerf/AnomalyDetection
- CCT/CCT_1_16_16_8, CCT/CCT_2_32_32_128_Opset20
- testTrainCCT/CCT1_Classifier_Training/CCT_1_16_16_8
- testFloatDemoTinyViT

---

### Siracusa-Tiled

**Workflow:** ci-platform-siracusa-tiled.yml
**Runner:** testRunner_tiled_siracusa.py
**Simulator:** GVSoC
**Cores:** 8
**Test Count:** 50+

**Jobs:**
1. **siracusa-kernels-tiled-singlebuffer-L2** (27 tests)
2. **siracusa-kernels-tiled-doublebuffer-L2** (24 tests)
3. **siracusa-models-tiled-singlebuffer-L2** (13 tests)
4. **siracusa-models-tiled-singlebuffer-L3** (9 tests)
5. **siracusa-models-tiled-doublebuffer-L3** (11 tests)

**Key Model Tests:**
- simpleRegression, miniMobileNet, miniMobileNetv2
- Attention, Transformer
- microLlama/microLlama1, microLlama/microLlama8, microLlama/microLlama8_parallel
- MLPerf/*, CCT/*
- testTrainCCT/CCT1_Classifier_Training/*
- testFloatDemoTinyViT

---

### Siracusa-Neureka-Tiled (Accelerator)

**Workflow:** ci-platform-siracusa-neureka-tiled.yml
**Runner:** testRunner_tiled_siracusa_w_neureka.py
**Simulator:** GVSoC
**Cores:** 8
**Test Count:** 11

**Kernel Tests:**
- testRequantizedLinear
- testPointwise
- testPointwiseConvBNReLU
- testPointwiseUnsignedWeights

**Model Tests:**
- miniMobileNet
- Attention
- Transformer
- microLlama/microLlama1

**Configurations:**
- With/without Neureka weight memory (wmem)
- Single/double buffering
- L2/L3 memory levels

---

### SoftHier

**Workflow:** ci-platform-softhier.yml
**Runner:** testRunner_softhier.py
**Toolchain:** GCC
**Test Count:** 1

**Tests:** Adder

---

### Chimera

**Workflow:** ci-platform-chimera.yml
**Runner:** testRunner_chimera.py
**Simulator:** GVSoC
**Test Count:** 1

**Tests:** Adder

---

## Component to Test Mapping

### TileConstraints

When modifying a TileConstraint, run all tiled tests that use that constraint.

| TileConstraint | Tests to Run |
|----------------|--------------|
| Conv2DTileConstraint | testFloat2DConvolution*, test2DRequantizedConv, miniMobileNet, CCT/*, testFloatDemoTinyViT |
| DWConv2DTileConstraint | testFloat2DDWConvolution*, testRequantizedDWConv, miniMobileNet*, testFloatDemoTinyViT |
| MatMulTileConstraint | testFloatMatmul, testMatMul, microLlama/*, Attention |
| GEMMTileConstraint | testFloatGEMM*, testRQGEMM*, testRequantizedLinear |
| MaxPoolTileConstraint | testFloatMaxPool, simpleRegression |
| LayernormTileConstraint | testFloatLayerNorm, CCT/*, testFloatDemoTinyViT |
| ConcatTileConstraint | testConcat, microLlama/* |
| TransposeTileConstraint | testFloatTranspose |
| iSoftmaxTileConstraint | iSoftmax, TestiSoftmaxLarge, microLlama/* |
| ReduceMeanTileConstraint | testReduceMean, testFloatDemoTinyViT |
| NeurekaPWConv2DTileConstraint | testPointwise*, miniMobileNet (Neureka) |
| NeurekaDenseConv2DTileConstraint | testRequantizedLinear (Neureka) |
| NeurekaDWConv2DTileConstraint | testPointwiseUnsignedWeights (Neureka) |

### Templates

When modifying a Template, run tests that use that template binding.

| Template Location | Affected Tests |
|-------------------|----------------|
| Generic/Templates/AddTemplate.py | All Add tests |
| Generic/Templates/ConvTemplate.py | All Conv tests |
| PULPOpen/Templates/FloatConvTemplate.py | Siracusa float Conv tests |
| PULPOpen/Templates/GEMMTemplate.py | Siracusa GEMM tests |
| Snitch/Templates/*.py | Snitch tests |
| Neureka/Templates/*.py | Neureka tests |

### Parsers/TypeCheckers/Bindings

Changes to these components affect all tests using the corresponding operator on the affected platform.

### Core Framework

| Component | Affected Tests |
|-----------|----------------|
| DeeployTypes.py | ALL tests |
| AbstractDataTypes.py | ALL tests |
| TilingExtension/*.py | ALL tiled tests |
| MemoryLevelExtension/*.py | ALL tests with memory levels |
| CommonExtensions/OptimizationPasses/*.py | Tests using that optimization |

---

## Test Category Index

### Unit Tests (70 tests)
Single operator tests for correctness validation.

**Run for:** Operator implementation changes, template fixes

### Integration Tests (7 tests)
Multi-operator combinations testing data flow.

**Run for:** Binding changes, optimization pass changes

### Model Tests (42 tests)
Full neural network models for end-to-end validation.

**Run for:** Major framework changes, memory management changes

---

## Change Impact Analysis Guide

### Quick Reference

| What Changed | Tests to Run |
|--------------|--------------|
| `Deeploy/DeeployTypes.py` | ALL tests |
| `Deeploy/TilingExtension/*` | ALL tiled tests |
| `Deeploy/Targets/Generic/*` | All Generic + inherited platform tests |
| `Deeploy/Targets/PULPOpen/*` | Siracusa tests |
| `Deeploy/Targets/Snitch/*` | Snitch tests |
| `Deeploy/Targets/Neureka/*` | Neureka tests |
| `TargetLibraries/PULPOpen/*` | Siracusa tests |
| `TargetLibraries/Generic/*` | Generic tests |

### Step-by-Step Process

1. **Identify changed files**
2. **Map to component type:**
   - Operator (Parser/TypeChecker/Binding/Template) → Operator tests
   - TileConstraint → Tiled tests using that constraint
   - Platform code → Platform tests
   - Core framework → Broad test coverage
3. **Select platforms:** Run on platforms where changes apply
4. **Select configurations:** Include tiled/non-tiled, different L1 sizes
5. **Run tests:** Use appropriate testRunner script

### Example Scenarios

**Scenario 1:** Modified `Conv2DTileConstraint.py`
```bash
# Run Siracusa tiled Conv tests
cd DeeployTest
python testRunner_tiled_siracusa.py -t Tests/testFloat2DConvolution --cores=8 --l1=2000
python testRunner_tiled_siracusa.py -t Tests/miniMobileNet --cores=8 --l1=6000
python testRunner_tiled_siracusa.py -t Tests/testFloatDemoTinyViT --cores=8 --l1=4000
```

**Scenario 2:** Modified `PULPOpen/Templates/GEMMTemplate.py`
```bash
# Run all GEMM tests on Siracusa
python testRunner_siracusa.py -t Tests/testFloatGEMM --cores=8
python testRunner_tiled_siracusa.py -t Tests/testFloatGEMM --cores=8 --l1=8000
python testRunner_tiled_siracusa.py -t Tests/testRQGEMM --cores=8 --l1=5000
```

**Scenario 3:** Modified `Deeploy/DeeployTypes.py`
```bash
# Run unit tests first
pytest testTypes.py
pytest testTilerExtension.py

# Then platform tests
python testRunner_generic.py -t Tests/Adder
python testRunner_siracusa.py -t Tests/Adder --cores=8
```

---

## Appendix: Test to Platform Matrix

| Test | Generic | CortexM | MemPool | Snitch | Siracusa | Siracusa-Tiled | Neureka |
|------|---------|---------|---------|--------|----------|----------------|---------|
| Adder | ✓ | ✓ | ✓ | ✓ | ✓ | - | - |
| testMatMul | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| testFloatGEMM | ✓ | - | - | - | ✓ | ✓ | - |
| testFloat2DConvolution | ✓ | - | - | - | ✓ | ✓ | - |
| iSoftmax | ✓ | - | - | ✓ | ✓ | ✓ | - |
| miniMobileNet | ✓ | - | ✓ | - | ✓ | ✓ | ✓ |
| testFloatDemoTinyViT | ✓ | - | - | - | ✓ | ✓ | - |
| testPointwise | - | - | - | - | - | - | ✓ |
| Attention | - | - | - | - | ✓ | ✓ | ✓ |
| microLlama/* | - | - | - | - | - | ✓ | ✓ |

---

## Files Reference

- **CI Mapping:** `/docs/ci_test_mapping.json`
- **Operator Mapping:** `/docs/test_operator_mapping.json`
- **This Document:** `/docs/CI_TEST_DEPENDENCIES.md`
