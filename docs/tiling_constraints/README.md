# Tiling Constraints Documentation

This directory contains comprehensive documentation for all tiling constraints used in Deeploy's automatic tiling system.

## Overview

Tiling constraints define the rules that govern how tensors can be split into smaller tiles for processing on memory-constrained embedded systems. Each constraint specifies:

- **Geometrical Constraints**: Mathematical relationships between tensor dimensions
- **Policy Constraints**: Optimization hints and algorithm requirements
- **Tiling Schedules**: How to compute input tiles from output tiles
- **Replacements**: Variables that are substituted with tile-specific values

## Documentation Index

### Generic Platform (Base Constraints)

| Operator | File | Description |
|----------|------|-------------|
| Binary Operations | [BinaryOp_Generic.md](BinaryOp_Generic.md) | Element-wise Add, Mul |
| Unary Operations | [UnaryTileConstraint_Generic.md](UnaryTileConstraint_Generic.md) | ReLU, Sigmoid, etc. |
| Concat | [ConcatTileConstraint_Generic.md](ConcatTileConstraint_Generic.md) | Tensor concatenation |
| Transpose | [TransposeTileConstraint_Generic.md](TransposeTileConstraint_Generic.md) | Dimension permutation |
| NOP | [NOPTileConstraint_Generic.md](NOPTileConstraint_Generic.md) | Pass-through operations |

### PULPOpen Platform

| Operator | File | Description |
|----------|------|-------------|
| Conv2D | [Conv2D_PULPOpen.md](Conv2D_PULPOpen.md) | 2D Convolution (Float & RQS) |
| DWConv2D | [DWConv2D_PULPOpen.md](DWConv2D_PULPOpen.md) | Depthwise Convolution |
| GEMM | [GEMM_PULPOpen.md](GEMM_PULPOpen.md) | General Matrix Multiply |
| MatMul | [MatMul_PULPOpen.md](MatMul_PULPOpen.md) | Matrix Multiplication |
| MaxPool | [MaxPool_PULPOpen.md](MaxPool_PULPOpen.md) | Max Pooling |
| Layernorm | [Layernorm_PULPOpen.md](Layernorm_PULPOpen.md) | Layer Normalization |
| ReduceMean | [ReduceMean_PULPOpen.md](ReduceMean_PULPOpen.md) | Mean Reduction |

### Snitch Platform

| Operator | File | Description |
|----------|------|-------------|
| GEMM | [GEMM_Snitch.md](GEMM_Snitch.md) | General Matrix Multiply |
| iSoftmax | [iSoftmax_Snitch.md](iSoftmax_Snitch.md) | Integer Softmax |

### Neureka Platform (Accelerator)

| Operator | File | Description |
|----------|------|-------------|
| Pointwise Conv | [PointwiseConv_Neureka.md](PointwiseConv_Neureka.md) | 1x1 Convolution |
| Dense Conv | [DenseConv_Neureka.md](DenseConv_Neureka.md) | 3x3 Dense Convolution |
| Depthwise Conv | [DepthwiseConv_Neureka.md](DepthwiseConv_Neureka.md) | 3x3 Depthwise Convolution |

## Template

New documentation should follow the structure in [TEMPLATE.md](TEMPLATE.md).

## Key Concepts

### Constraint Types

1. **Hard Constraints**: Must be satisfied for correctness (e.g., inner dimension non-tiling in GEMM)
2. **Performance Hints**: Optimization suggestions with priorities (e.g., SIMD alignment)

### Common Patterns

- **Spatial tiling**: Splitting height/width dimensions (Conv2D, MaxPool)
- **Channel tiling**: Splitting channel dimensions (some operations support this)
- **Reduction dimension preservation**: Inner dimensions cannot be tiled (GEMM, MatMul, Layernorm)

### Reading the Documentation

Each constraint document includes:

1. **Propositions**: Logical expressions in the form `variable == expression` or `variable >= minimum`
2. **Source references**: Line numbers in the source Python files
3. **Visual diagrams**: ASCII art showing tiling patterns
4. **Examples**: Concrete numerical examples with resulting tile configurations

## Contributing

When adding new tiling constraints:

1. Copy [TEMPLATE.md](TEMPLATE.md) to `<OperatorName>_<Platform>.md`
2. Fill in all sections based on the source constraint file
3. Update this README index
4. Run `make format` to ensure consistent formatting

## References

- Source code: `Deeploy/Targets/*/TileConstraints/`
- Tiler configuration: `Deeploy/Targets/*/Tiler.py`
- Tiling framework: `Deeploy/TilingExtension/`
