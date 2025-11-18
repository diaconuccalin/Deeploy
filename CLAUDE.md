# CLAUDE.md - AI Assistant Guide for Deeploy

> **Version:** 0.2.1
> **Last Updated:** 2025-11-18
> **Project:** Deeploy - DNN Compiler for Heterogeneous SoCs

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Architecture](#codebase-architecture)
3. [Directory Structure](#directory-structure)
4. [Key Components & Design Patterns](#key-components--design-patterns)
5. [Development Workflows](#development-workflows)
6. [Code Conventions & Style Guide](#code-conventions--style-guide)
7. [Testing Strategy](#testing-strategy)
8. [Common Tasks for AI Assistants](#common-tasks-for-ai-assistants)
9. [Important Files Reference](#important-files-reference)
10. [Build System & Toolchains](#build-system--toolchains)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Tips for AI Assistants](#tips-for-ai-assistants)

---

## Project Overview

**Deeploy** is an ONNX-to-C compiler that generates low-level optimized C code for multi-cluster, heterogeneous System-on-Chips (SoCs). It is developed as part of the PULP (Parallel Ultra-Low-Power) project, a joint effort between ETH Zurich and the University of Bologna.

### Key Characteristics

- **Input:** ONNX neural network models
- **Output:** Optimized C code for embedded systems
- **Target:** Heterogeneous SoCs with multiple compute engines and memory hierarchies
- **Approach:** Bottom-up compiler design with fine-grained hardware modeling
- **Language:** ~37,000 lines of Python (303 files)
- **License:** Apache 2.0 (most files), MIT (some scripts), CC BY 4.0 (docs/tests)

### Supported Target Platforms

1. **Generic CPU** - Host machine (development/testing)
2. **CortexM Processors** - ARM Cortex-M4 (QEMU simulator)
3. **MemPool + ITA** - Multi-core RISC-V cluster (Banshee simulator)
4. **Siracusa** - PULP cluster with Neureka accelerator (GVSoC simulator)
5. **Snitch Cluster** - RISC-V cluster (GVSoC simulator)
6. **SoftHier** - Software-managed memory hierarchy (GVSoC simulator)
7. **Chimera** - Heterogeneous PULP SoC (GVSoC simulator)
8. **PULPOpen** - PULP platform

### Core Capabilities

- ONNX operator compilation to optimized C kernels
- Multi-level memory hierarchy management (L1/L2/L3)
- Automatic tiling for large tensors
- Multi-engine heterogeneous execution
- Asynchronous DMA support
- Quantization (int8, int16) and floating-point (fp16, fp32)
- Transformer and CNN model support

---

## Codebase Architecture

### High-Level Structure

```
Deeploy/                    # Main Python package
├── DeeployTypes.py         # Core compilation framework (3,600+ lines)
├── AbstractDataTypes.py    # Type system foundation (560 lines)
├── Logging.py              # Centralized logging
├── Targets/                # Platform-specific implementations
│   ├── Generic/
│   ├── CortexM/
│   ├── MemPool/
│   ├── Snitch/
│   ├── Siracusa/
│   ├── Neureka/
│   ├── PULPOpen/
│   ├── Chimera/
│   └── SoftHier/
├── CommonExtensions/       # Shared optimization passes
├── TilingExtension/        # Tiling infrastructure (52,000+ lines)
├── EngineExtension/        # Multi-engine support
├── MemoryLevelExtension/   # Memory hierarchy modeling
└── FutureExtension/        # Async/concurrency support

DeeployTest/                # Test infrastructure
├── Tests/                  # 90+ test networks
├── Platforms/              # Platform configurations
├── testUtils/              # Test execution framework
└── testRunner_*.py         # Platform-specific test runners

TargetLibraries/            # C runtime libraries
├── Generic/                # Reference implementations
├── MemPool/                # MemPool-optimized kernels
├── Snitch/                 # Snitch-optimized kernels
├── CMSIS/                  # ARM CMSIS-NN integration
└── PULPOpen/               # PULP platform kernels
```

### Compilation Pipeline

```
ONNX Model
    ↓
1. PARSE          → NodeParser extracts attributes per operator
    ↓
2. LOWER          → TopologyOptimizer applies graph transformations
    ↓
3. TYPE CHECK     → NodeTypeChecker validates type compatibility
    ↓
4. BIND           → NodeBinding selects code template based on types
    ↓
5. MEMORY ALLOC   → MemoryLevelExtension assigns to memory hierarchy
    ↓
6. TILING         → TilingExtension generates tile loops (optional)
    ↓
7. CODE TRANSFORM → CodeTransformationPass applies optimizations
    ↓
8. CODE GENERATE  → NodeTemplate renders C code via Mako
    ↓
Optimized C Code
```

---

## Directory Structure

### Top-Level Organization

```
.
├── .github/                # CI/CD workflows (30+ workflow files)
│   └── workflows/          # Platform-specific CI, linting, Docker builds
├── cmake/                  # CMake toolchain files
│   ├── common.cmake
│   ├── mempool/
│   ├── snitch/
│   └── simulation.cmake
├── Container/              # Docker configuration
├── Deeploy/                # Main Python source code
├── DeeployTest/            # Test suite
├── docs/                   # Sphinx documentation
│   ├── _static/
│   ├── tutorials/
│   ├── install.md
│   └── structure.md
├── LICENSES/               # License files
├── scripts/                # Utility scripts
├── TargetLibraries/        # C runtime libraries per platform
├── toolchain/              # Toolchain build scripts
├── CMakeLists.txt          # Root CMake configuration
├── Makefile                # Toolchain/emulator build automation
├── pyproject.toml          # Python package configuration
├── setup.py                # Python package setup
├── .pre-commit-config.yaml # Pre-commit hooks
├── .clang-format           # C/C++ formatting (LLVM style, 80 cols)
├── .style.yapf             # Python formatting (Google style, 120 cols)
├── .isort.cfg              # Python import sorting
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
└── README.md               # Project documentation
```

### Deeploy/ Package Structure

```
Deeploy/
├── __init__.py
├── AbstractDataTypes.py       # Type system: BaseType, Immediate, Pointer, Struct
├── DeeployTypes.py            # 27 core classes for compilation
├── Logging.py                 # DEFAULT_LOGGER with color support
│
├── CommonExtensions/          # Shared functionality across targets
│   ├── CodeTransformationPasses/
│   │   ├── MemoryAllocation.py     # Memory management codegen
│   │   ├── Closure.py              # Closure generation
│   │   ├── PrintInputs.py          # Debug instrumentation
│   │   └── CycleMeasurement.py     # Performance profiling
│   ├── NetworkDeployers/
│   │   ├── NetworkDeployerWrapper.py  # Extension decorator pattern
│   │   └── SignPropDeployer.py        # Signedness propagation
│   ├── OptimizationPasses/
│   │   ├── PassClasses.py          # Base optimization framework
│   │   ├── Matchers.py             # Graph pattern matching
│   │   ├── BindingsOptimizationPasses/
│   │   └── TopologyOptimizationPasses/
│   └── TypeCheckers/
│       └── SignPropTypeChecker.py
│
├── EngineExtension/           # Multi-engine heterogeneous execution
│   ├── NetworkDeployers/
│   │   └── EngineColoringDeployer.py  # Assign ops to engines
│   └── OptimizationPasses/
│
├── MemoryLevelExtension/      # Memory hierarchy management
│   ├── MemoryLevels.py        # MemoryLevel, MemoryHierarchy classes
│   ├── NetworkDeployers/
│   │   └── MemoryLevelDeployer.py
│   └── OptimizationPasses/
│
├── TilingExtension/           # Operator tiling (52,000+ lines)
│   ├── TilerExtension.py      # Main Tiler class (1,800+ lines)
│   ├── TilerModel.py          # Symbolic constraint modeling
│   ├── MemoryScheduler.py     # Tile scheduling (Tetris, MiniMalloc)
│   ├── MemoryConstraints.py   # Tile size constraints
│   ├── TileConstraint.py      # Tiling applicability
│   ├── AsyncDma.py            # Async DMA codegen
│   └── TilingCodegen.py       # Tiled loop generation
│
├── FutureExtension/           # Asynchronous execution support
│   ├── Future.py              # Future abstraction
│   ├── Bindings/
│   └── CodeTransformationPasses/
│
└── Targets/                   # Platform implementations
    ├── Generic/
    │   ├── Platform.py        # GenericMapping, buffer classes
    │   ├── Parsers.py         # NodeParser subclasses
    │   ├── Bindings.py        # NodeBinding definitions
    │   ├── Layers.py          # ONNXLayer subclasses
    │   ├── TypeCheckers.py    # NodeTypeChecker subclasses
    │   ├── Deployer.py        # Platform-specific NetworkDeployer
    │   └── Templates/         # Mako code generation templates
    │       ├── AddTemplate.py
    │       ├── ConvTemplate.py
    │       ├── MatMulTemplate.py
    │       ├── GEMMTemplate.py
    │       └── ... (30+ templates)
    │
    └── [CortexM, MemPool, Snitch, Siracusa, etc.]
        └── (Same structure as Generic/)
```

---

## Key Components & Design Patterns

### 1. Type System (`AbstractDataTypes.py`)

**Foundation for C type compatibility:**

```python
BaseType                    # Abstract base for all C types
├── Immediate               # Literal values
│   ├── IntegerImmediate    # int8_t, uint32_t, etc.
│   └── FloatImmediate      # float16, float32, bfloat16, float64
├── Pointer                 # C pointer with reference tracking
└── Struct                  # Packed struct definitions
```

**Key Classes:**
- `BaseType`: Abstract base with `typeString`, `referencedType`
- `Immediate`: Base for scalar/array literals
- `Pointer`: Created via `PointerClass(referencedType)`
- `Struct`: Created via `StructClass(name, fieldMapping)`

**Usage:** All tensor buffers, weights, and intermediate values use these types for C code generation.

---

### 2. Core Compilation Framework (`DeeployTypes.py`)

**27 Major Classes organized into 6 categories:**

#### a. Code Generation

**`NodeTemplate`**
- Wraps Mako templates with expression hoisting
- Methods: `__init__(templateStr)`, `generate(operatorRepresentation)`
- Renders C code from operator representation

**`CodeSnippet`**
- Pairs `NodeTemplate` with `OperatorRepresentation`
- Properties: `template`, `operatorRepresentation`, `name`

**`CodeGenVerbosity`**
- Configuration flags for profiling/instrumentation
- Fields: `default`, `minimal`, `minimal_with_perf`

#### b. Memory Management

**`VariableBuffer`**
- Represents input/output tensors
- Properties: `name`, `shape`, `dtype`, `memoryLevel`, `initTemplate`, `allocTemplate`

**`TransientBuffer`** (extends `VariableBuffer`)
- Temporary buffers created/destroyed per node
- Additional property: `_users` (reference counting)

**`ConstantBuffer`** (extends `VariableBuffer`)
- Immutable weights/biases
- Additional property: `values` (numpy array)

**`StructBuffer`** (extends `VariableBuffer`)
- Structured data containers
- Additional property: `structTypeDict` (field mappings)

#### c. Compilation Pipeline

**`NetworkContext`**
- Central registry for tensors, constants, buffers
- Key methods:
  - `lookup(name) → VariableBuffer`
  - `hoistConstant(value, name) → str` (registers constant)
  - `registerBuffer(buffer)` (adds to global context)
  - `annotateType(name, type)` (type annotation)
- Properties: `globalObjects`, `localObjects`, `typeDict`, `lookup_cache`

**`NodeParser`** (Abstract)
- Analyzes ONNX node attributes
- Methods:
  - `parseNode(node) → Dict` (extract attributes)
  - `parseNodeCtxt(node, ctxt) → OperatorRepresentation` (process IO)
- Subclassed per operator: `Conv2DParser`, `MatMulParser`, etc.

**`NodeTypeChecker`** (Abstract)
- Validates type compatibility
- Method: `typeCheck(operatorRepresentation) → bool`
- Subclassed per operator: `ConvChecker`, `MatMulChecker`, etc.

**`NodeBinding`**
- Maps type signatures to code templates
- Properties:
  - `parser`: NodeParser instance
  - `checker`: NodeTypeChecker instance
  - `template`: NodeTemplate instance
  - `signatureConstraint`: Type signature (e.g., `[int8, int8] -> int32`)
- Method: `bind(node, ctxt) → CodeSnippet`

**`NodeMapper`**
- Combines Parser, TypeChecker, Binding
- Represents one implementation strategy for an operator

**`ONNXLayer`**
- Wraps ONNX operator with multiple mappers
- Properties: `nodeName`, `mappers` (list of NodeMapper)
- Method: `computeOps()` (operation count for profiling)

#### d. Optimization Framework

**`TopologyOptimizationPass`** (Abstract)
- Graph-level transformations before compilation
- Method: `apply(graph) → graph`
- Examples: quantization, operator fusion, constant folding

**`TopologyOptimizer`**
- Pipeline of topology passes
- Method: `optimize(graph, passes) → graph`

**`NetworkOptimizationPass`** (Abstract)
- Extended passes with network context
- Method: `apply(graph, ctxt) → graph`

**`CodeTransformationPass`** (Abstract)
- Post-binding code transformations
- Method: `apply(ctxt, codeSnippets) → codeSnippets`
- Examples: memory allocation, closure generation, profiling

**`CodeTransformation`**
- Pipeline for code-level optimizations
- Method: `transform(ctxt, codeSnippets, passes) → codeSnippets`

#### e. Deployment Abstractions

**`DeploymentEngine`**
- Represents a compute engine (CPU, accelerator)
- Properties: `name`, `operatorMapping` (ONNX op → ONNXLayer)
- Method: `canExecute(node) → bool`

**`DeploymentPlatform`**
- Complete system model
- Properties:
  - `engines`: List of DeploymentEngine
  - `memoryHierarchy`: MemoryHierarchy instance
  - `bufferClasses`: Platform-specific buffer types

**`NetworkContainer`**
- Base container for graph, platform, types
- Properties: `graph`, `platform`, `inputTypes`

**`NetworkDeployer`** (extends `NetworkContainer`)
- Main compilation orchestrator
- Key methods:
  - `parse() → None` (parse all nodes)
  - `lower() → None` (apply topology optimizations)
  - `typeCheck() → None` (validate types)
  - `bind() → None` (select bindings)
  - `codeTransform() → None` (apply code optimizations)
  - `generateInferenceCode() → str` (render compute kernels)
  - `generateBufferAllocationCode() → str` (render memory mgmt)
  - `generateFunction() → str` (assemble complete C function)

#### f. Execution Model

**`ExecutionBlock`**
- Represents computation with scheduled nodes
- Properties: `nodes`, `schedule`, `memoryFootprint`

---

### 3. Design Patterns

#### Decorator/Wrapper Pattern (Extension Model)

Extensions augment `NetworkDeployer` via wrappers:

```
NetworkDeployer (base)
    ↓ wrapped by
SignPropDeployer
    ↓ wrapped by
MemoryLevelAwareDeployer
    ↓ wrapped by
TilerDeployerWrapper
    ↓ wrapped by
EngineColoringDeployerWrapper
```

**Implementation:**
- Base class: `NetworkDeployerWrapper`
- Uses `__getattr__` to delegate method calls to wrapped deployer
- Each wrapper overrides specific methods (e.g., `bind()`, `codeTransform()`)

**Key File:** `Deeploy/CommonExtensions/NetworkDeployers/NetworkDeployerWrapper.py`

#### Strategy Pattern (Operator Implementation)

Multiple `NodeMapper` strategies per operator, one per type signature:

```python
ONNXLayer("Add", mappers=[
    NodeMapper(parser, int8_checker, int8_template),    # int8 + int8 → int8
    NodeMapper(parser, fp32_checker, fp32_template),    # fp32 + fp32 → fp32
    NodeMapper(parser, mixed_checker, mixed_template),  # int8 + int16 → int32
])
```

**Selection:** During binding, the deployer tries each mapper's type checker until one succeeds.

#### Pipeline Pattern (Compilation Stages)

Each stage transforms the representation:

1. **Parse** → ONNX nodes → `OperatorRepresentation` (with attributes)
2. **Lower** → Graph → Optimized graph (operator fusion, constant folding)
3. **TypeCheck** → `OperatorRepresentation` → Validated representation
4. **Bind** → Validated representation → `CodeSnippet` (template + repr)
5. **CodeTransform** → `CodeSnippet` list → Optimized snippets
6. **Generate** → `CodeSnippet` → C code string

#### Registry Pattern (NetworkContext)

Central registry for tensors, constants, buffers:

```python
ctxt = NetworkContext()
ctxt.lookup("input_tensor")           # Retrieve VariableBuffer
ctxt.hoistConstant(weights, "w0")     # Register constant
ctxt.registerBuffer(transient_buf)    # Register buffer
ctxt.annotateType("output", int32_t)  # Annotate type
```

#### Template Method Pattern

Base classes define algorithm skeleton; subclasses override steps:

```python
class NodeParser(ABC):
    def parse(self, node):         # Template method
        attrs = self.parseNode(node)        # Override in subclass
        repr = self.parseNodeCtxt(node, ctxt)  # Override in subclass
        return repr
```

---

## Development Workflows

### 1. Setting Up Development Environment

#### Option A: Docker (Recommended)

```bash
# Pull Docker image
docker pull ghcr.io/pulp-platform/deeploy:main

# Create and start container
docker run -it --name deeploy_main -v $(pwd):/app/Deeploy \
    ghcr.io/pulp-platform/deeploy:main

# Install Deeploy in editable mode
cd Deeploy
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com
```

#### Option B: Local Installation

```bash
# Clone repository with submodules
git clone https://github.com/pulp-platform/Deeploy.git
cd Deeploy
git submodule update --init --recursive

# Install Python package
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com

# Build toolchains (optional, for embedded targets)
make toolchain

# Build emulators (optional)
make emulators
```

### 2. Pre-Commit Hooks

**Setup:**

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks (default stage: pre-push)
pre-commit install --hook-type pre-push
```

**Hooks configured:**
- **SPDX license headers** (via `reuse` tool)
- **Python formatting** (yapf, isort, autoflake)
- **C/C++ formatting** (clang-format)
- **YAML linting** (yamllint)
- **Trailing whitespace check**
- **Large file check**

**Manual execution:**

```bash
# Format all files
make format

# Lint without modifying
make lint
```

### 3. Running Tests

#### Platform-Specific Test Runners

```bash
cd DeeployTest

# Generic CPU
python testRunner_generic.py -t Tests/Adder

# CortexM
python testRunner_cortexm.py -t Tests/Adder

# MemPool
python testRunner_mempool.py -t Tests/Adder

# Siracusa
python testRunner_siracusa.py -t Tests/Adder --cores=8

# Snitch
python testRunner_snitch.py -t Tests/Adder --cores=9

# SoftHier
python testRunner_softhier.py -t Tests/Adder --toolchain=GCC

# Chimera
python testRunner_chimera.py -t Tests/Adder
```

#### Tiled Test Runners

```bash
# Siracusa with tiling
python testRunner_tiled_siracusa.py -t Tests/testMatMul --cores=8 --l1=16000

# Siracusa with Neureka accelerator
python testRunner_tiled_siracusa_w_neureka.py -t Tests/testConv --cores=8

# Snitch with tiling
python testRunner_tiled_snitch.py -t Tests/testMatMul --cores=9 --l1=64000
```

#### Unit Tests

```bash
# Type system tests
pytest testTypes.py

# Tiling extension tests
pytest testTilerExtension.py

# Memory level extension tests
pytest testMemoryLevelExtension.py

# MVP tests
pytest testMVP.py
```

### 4. Code Generation Workflow

**Example:** Compile ONNX model for Siracusa platform

```python
from Deeploy.Targets.Siracusa.Platform import SiracusaPlatform
from Deeploy.Targets.Siracusa.Deployer import SiracusaDeployer
import onnx

# Load ONNX model
model = onnx.load("model.onnx")
graph = model.graph

# Create platform with 8 cores
platform = SiracusaPlatform(cores=8)

# Create deployer
deployer = SiracusaDeployer(graph, platform, inputTypes={'input': int8_t})

# Run compilation pipeline
deployer.parse()
deployer.lower()
deployer.typeCheck()
deployer.bind()
deployer.codeTransform()

# Generate C code
c_code = deployer.generateFunction()
print(c_code)
```

### 5. Adding a New Operator

**Steps:**

1. **Create Parser** in `Deeploy/Targets/Generic/Parsers.py`:

```python
class MyOpParser(NodeParser):
    def parseNode(self, node):
        return {
            'param1': node.attribute['param1'],
            'param2': node.attribute['param2'],
        }

    def parseNodeCtxt(self, node, ctxt):
        inputs = [ctxt.lookup(inp.name) for inp in node.input]
        outputs = [ctxt.lookup(out.name) for out in node.output]
        return OperatorRepresentation('MyOp', node.name, inputs, outputs, self.parseNode(node))
```

2. **Create TypeChecker** in `Deeploy/Targets/Generic/TypeCheckers.py`:

```python
class MyOpChecker(NodeTypeChecker):
    def typeCheck(self, operatorRepresentation):
        inputs = operatorRepresentation.inputs
        outputs = operatorRepresentation.outputs

        # Check input types match
        assert inputs[0].dtype == int8_t
        assert outputs[0].dtype == int8_t
        return True
```

3. **Create Template** in `Deeploy/Targets/Generic/Templates/MyOpTemplate.py`:

```python
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
void ${name}(${data_in.name}_t *${data_in.name}, ${data_out.name}_t *${data_out.name},
             int ${param1}, int ${param2}) {
    // Implementation
    for (int i = 0; i < ${size}; i++) {
        ${data_out.name}[i] = compute(${data_in.name}[i], ${param1}, ${param2});
    }
}
""")
```

4. **Create Binding** in `Deeploy/Targets/Generic/Bindings.py`:

```python
from Deeploy.CommonExtensions.DataTypes import int8_t

MyOpBindings = [
    NodeBinding(
        parser=MyOpParser(),
        checker=MyOpChecker(),
        template=referenceTemplate,
        signatureConstraint=([int8_t], [int8_t])
    )
]
```

5. **Register Layer** in `Deeploy/Targets/Generic/Layers.py`:

```python
class MyOpLayer(ONNXLayer):
    def __init__(self, mappers):
        super().__init__("MyOp", mappers)

    def computeOps(self, operatorRepresentation):
        # Compute operation count
        return operatorRepresentation.inputs[0].size
```

6. **Update Platform Mapping** in `Deeploy/Targets/Generic/Platform.py`:

```python
GenericMapping = {
    # ... existing mappings
    'MyOp': MyOpLayer([MyOpMapper]),
}
```

### 6. Submitting Pull Requests

**Process:**

1. **Branch from `devel`:**

```bash
git checkout devel
git pull origin devel
git checkout -b feature/my-new-feature
```

2. **Make changes and test:**

```bash
# Run relevant tests
cd DeeployTest
python testRunner_generic.py -t Tests/MyOpTest

# Run linting
cd ..
make lint
```

3. **Update CHANGELOG.md:**

```markdown
## Unreleased (Planned Release Target: vx.x.x)

### List of Pull Requests
- Add MyOp operator support [#123](https://github.com/pulp-platform/Deeploy/pull/123)

### Added
- Added MyOp operator with int8 support
- Added regression test for MyOp in Tests/MyOpTest
```

4. **Commit with descriptive message:**

```bash
git add .
git commit -m "Add MyOp operator with int8 support

- Implemented parser, type checker, and template for MyOp
- Added regression test
- Updated CHANGELOG.md"
```

5. **Push and create PR:**

```bash
git push origin feature/my-new-feature
# Open PR on GitHub against `devel` branch
```

**PR Guidelines:**
- Use **DRAFT:** prefix for work-in-progress
- Use **REFACTOR:** prefix for refactoring PRs
- Reference issues in description (e.g., "Fixes #42")
- Include proof-of-concept or regression test
- Respond to review feedback with actionable items

---

## Code Conventions & Style Guide

### Python Style

**Formatter:** YAPF (Google style)

**Configuration (.style.yapf):**
- `based_on_style = google`
- `column_limit = 120`
- `split_before_logical_operator = true`
- `spaces_around_default_or_named_assign = true`

**Import Sorting:** isort

**Configuration (.isort.cfg):**
- `line_length = 120`
- `multi_line_output = 2`
- `include_trailing_comma = false`

**Auto-formatting:**

```bash
# Format Python files
yapf --in-place --parallel --recursive Deeploy/

# Sort imports
isort Deeploy/

# Or use make target
make format
```

**Naming Conventions:**
- Classes: `PascalCase` (e.g., `NetworkDeployer`, `NodeParser`)
- Functions/methods: `camelCase` (e.g., `parseNode`, `typeCheck`)
- Private methods: `_camelCase` (e.g., `_selectEngine`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_LOGGER`)
- Modules: `lowercase` or `PascalCase` for class modules

### C/C++ Style

**Formatter:** clang-format (LLVM style)

**Configuration (.clang-format):**
- `BasedOnStyle: LLVM`
- `ColumnLimit: 80`

**Auto-formatting:**

```bash
# Format C files
find TargetLibraries/ -name "*.c" -o -name "*.h" | xargs clang-format -i

# Or use make target
make format
```

**Naming Conventions:**
- Functions: `snake_case` (e.g., `conv2d_kernel`, `allocate_buffer`)
- Variables: `snake_case`
- Macros: `UPPER_SNAKE_CASE`
- Structs: `snake_case_t` suffix (e.g., `network_context_t`)

### SPDX License Headers

**All files require SPDX headers:**

```python
# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
```

**Add header using reuse tool:**

```bash
reuse annotate --copyright "ETH Zurich and University of Bologna" \
               --license Apache-2.0 \
               myfile.py
```

### Documentation Standards

**Docstrings:** Google style

```python
def parseNode(self, node: onnx.NodeProto) -> Dict[str, Any]:
    """Parse ONNX node attributes.

    Args:
        node: ONNX node protobuf

    Returns:
        Dictionary of parsed attributes

    Raises:
        ValueError: If required attributes are missing
    """
    pass
```

**Inline comments:**
- Explain "why" not "what"
- Use `#` for single-line comments
- Keep comments up-to-date with code changes

---

## Testing Strategy

### Test Organization

```
DeeployTest/
├── Tests/                      # 90+ test networks
│   ├── Adder/                  # Simple element-wise add
│   ├── testMatMul/             # Matrix multiplication
│   ├── testConv/               # 2D convolution
│   ├── WaveFormer/             # Full transformer model
│   └── ...
├── Platforms/                  # Platform configurations
│   ├── Generic/
│   ├── Siracusa/
│   └── ...
├── testUtils/                  # Test framework
│   ├── testRunner.py           # Core TestRunner class
│   ├── codeGenerate.py         # Code generation utilities
│   ├── platformMapping.py      # Platform configuration
│   └── ...
└── testRunner_*.py             # Platform-specific runners
```

### Test Levels

1. **Unit Tests** (pytest)
   - `testTypes.py`: Type system validation
   - `testTilerExtension.py`: Tiling constraints
   - `testMemoryLevelExtension.py`: Memory hierarchy
   - `testComponentGraph.py`: Graph operations

2. **Integration Tests** (testRunner scripts)
   - End-to-end compilation
   - C code generation
   - Compilation with platform toolchain
   - Execution on simulator
   - Output validation against ONNX reference

3. **Regression Tests**
   - All tests in `Tests/` directory
   - Run on CI for each platform
   - Compare numerical output with ONNX runtime

### Test Execution

**Basic test:**

```bash
cd DeeployTest
python testRunner_generic.py -t Tests/Adder
```

**Test with options:**

```bash
# Siracusa with 8 cores, 16KB L1
python testRunner_siracusa.py -t Tests/testConv --cores=8 --l1=16000

# Snitch with tiling, 64KB L1
python testRunner_tiled_snitch.py -t Tests/testMatMul --cores=9 --l1=64000
```

**Test structure:**

```
Tests/MyTest/
├── network.onnx            # ONNX model
├── input.npy               # Input tensor (NumPy)
└── output.npy              # Expected output (from ONNX runtime)
```

**TestRunner workflow:**

1. Load ONNX model from `network.onnx`
2. Load input from `input.npy`
3. Run ONNX runtime to get reference output
4. Compile ONNX to C using Deeploy
5. Compile C code with platform toolchain
6. Run on simulator
7. Compare output with reference (tolerance: 1e-5 for float, exact for int)

### Writing New Tests

**Example:**

```python
import onnx
import numpy as np
from onnx import helper, TensorProto

# Create ONNX graph
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])

node = helper.make_node('MyOp', inputs=['input'], outputs=['output'], param1=5, param2=10)

graph = helper.make_graph([node], 'MyOpTest', [input_tensor], [output_tensor])
model = helper.make_model(graph, producer_name='deeploy-test')

# Save model
onnx.save(model, 'Tests/MyOpTest/network.onnx')

# Create input
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
np.save('Tests/MyOpTest/input.npy', input_data)

# Run test
# python testRunner_generic.py -t Tests/MyOpTest
```

---

## Common Tasks for AI Assistants

### 1. Adding Support for a New Target Platform

**Files to create:**

```
Deeploy/Targets/NewPlatform/
├── __init__.py
├── Platform.py          # NewPlatformMapping, buffer classes
├── Parsers.py           # Platform-specific parsers (if needed)
├── Bindings.py          # NodeBinding definitions
├── Layers.py            # ONNXLayer subclasses
├── TypeCheckers.py      # NodeTypeChecker subclasses
├── Deployer.py          # NewPlatformDeployer
└── Templates/           # Mako templates for code generation
    ├── AddTemplate.py
    ├── ConvTemplate.py
    └── ...

TargetLibraries/NewPlatform/
├── CMakeLists.txt       # Library build
├── inc/                 # Header files
│   └── kernels.h
└── src/                 # Kernel implementations
    ├── add.c
    ├── conv.c
    └── ...

DeeployTest/
└── testRunner_newplatform.py
```

**Steps:**

1. Study `Deeploy/Targets/Generic/` as reference
2. Create platform-specific templates in `Templates/`
3. Define operator mapping in `Platform.py`
4. Implement `Deployer.py` extending `NetworkDeployer`
5. Create C runtime library in `TargetLibraries/NewPlatform/`
6. Write test runner `testRunner_newplatform.py`
7. Add CI workflow in `.github/workflows/ci-platform-newplatform.yml`

### 2. Optimizing an Existing Operator

**Optimization types:**

- **Topology optimization:** Graph-level pattern matching and fusion
- **Binding optimization:** Better type signature selection
- **Code transformation:** Post-binding code optimizations

**Example: Fusing Conv + ReLU**

```python
# In Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/

class ConvReluFusionPass(TopologyOptimizationPass):
    def apply(self, graph):
        for node in graph.nodes:
            if node.op_type == 'Conv':
                # Find successor nodes
                successors = [n for n in graph.nodes if node.output[0] in n.input]
                if len(successors) == 1 and successors[0].op_type == 'Relu':
                    # Fuse Conv + ReLU
                    fused_node = self._fuse(node, successors[0])
                    graph.replace_nodes([node, successors[0]], fused_node)
        return graph
```

### 3. Debugging Generated C Code

**Techniques:**

1. **Enable debug instrumentation:**

```python
from Deeploy.CommonExtensions.CodeTransformationPasses import PrintInputs

deployer.codeTransform(passes=[PrintInputs()])
```

2. **Inspect generated code:**

```python
c_code = deployer.generateFunction()
print(c_code)
```

3. **Check intermediate representations:**

```python
# After binding
for snippet in deployer.codeSnippets:
    print(f"{snippet.name}: {snippet.operatorRepresentation}")
```

4. **Use visualization:**

```python
from DeeployTest.testUtils.graphDebug import visualizeGraph

visualizeGraph(graph, filename='debug_graph.png')
```

### 4. Investigating CI Failures

**Common failure types:**

- **Linting failures:** Run `make lint` locally
- **Test failures:** Check test logs in CI artifacts
- **Platform-specific failures:** Run relevant `testRunner_*.py` locally

**Debug steps:**

1. Check CI logs for error message
2. Reproduce locally with same test command
3. If platform-specific, use Docker image:

```bash
docker run -it --name deeploy_debug -v $(pwd):/app/Deeploy \
    ghcr.io/pulp-platform/deeploy:main
cd /app/Deeploy/DeeployTest
python testRunner_siracusa.py -t Tests/FailingTest --cores=8
```

### 5. Understanding Memory Hierarchy

**Memory level assignment:**

```python
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryLevel, MemoryHierarchy

# Define hierarchy
l1 = MemoryLevel(name='L1', size=64000, accessLatency=1)
l2 = MemoryLevel(name='L2', size=512000, accessLatency=10)
l3 = MemoryLevel(name='L3', size=8*1024*1024, accessLatency=100)

hierarchy = MemoryHierarchy([l1, l2, l3])

# Assign tensor to L1
tensor.memoryLevel = 'L1'
```

**With tiling:**

```python
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper

deployer = TilerDeployerWrapper(
    baseDeployer=SiracusaDeployer(graph, platform, inputTypes),
    memoryHierarchy=hierarchy,
    tileScheduler='Tetris'  # Options: 'Tetris', 'MiniMalloc'
)
```

### 6. Analyzing Performance

**Cycle counting:**

```python
from Deeploy.CommonExtensions.CodeTransformationPasses import CycleMeasurement

deployer.codeTransform(passes=[CycleMeasurement()])
c_code = deployer.generateFunction()
# Generated code will include cycle counters
```

**Profiling output:**

```bash
# Run with profiling enabled
python testRunner_siracusa.py -t Tests/testConv --cores=8 --profile

# Extract cycle counts
python profiling2csv.py --input TEST_SIRACUSA/Tests/testConv/profile.txt \
                        --output profile.csv
```

---

## Important Files Reference

### Core Compilation

| File | Lines | Purpose |
|------|-------|---------|
| `Deeploy/DeeployTypes.py` | 3,600+ | Core compilation framework (27 classes) |
| `Deeploy/AbstractDataTypes.py` | 560 | Type system foundation |
| `Deeploy/Logging.py` | 60 | Centralized logging |

### Extensions

| File | Lines | Purpose |
|------|-------|---------|
| `Deeploy/TilingExtension/TilerExtension.py` | 1,800+ | Main tiling orchestration |
| `Deeploy/TilingExtension/MemoryScheduler.py` | 1,000+ | Tile scheduling algorithms |
| `Deeploy/MemoryLevelExtension/MemoryLevels.py` | 150+ | Memory hierarchy modeling |
| `Deeploy/CommonExtensions/NetworkDeployers/NetworkDeployerWrapper.py` | 100 | Extension wrapper pattern |

### Platform Examples

| File | Lines | Purpose |
|------|-------|---------|
| `Deeploy/Targets/Generic/Platform.py` | 200 | Reference platform implementation |
| `Deeploy/Targets/Siracusa/Deployer.py` | 300 | Siracusa-specific deployer |
| `Deeploy/Targets/CortexM/Templates/GEMMTemplate.py` | 200 | CMSIS-NN GEMM template |

### Testing

| File | Lines | Purpose |
|------|-------|---------|
| `DeeployTest/testUtils/testRunner.py` | 250+ | Core test execution framework |
| `DeeployTest/testRunner_siracusa.py` | 40 | Siracusa test runner |
| `DeeployTest/testTilerExtension.py` | 250 | Tiling unit tests |

### Build System

| File | Lines | Purpose |
|------|-------|---------|
| `Makefile` | 700+ | Toolchain/emulator build automation |
| `CMakeLists.txt` | 330+ | Root CMake configuration |
| `pyproject.toml` | 50 | Python package metadata |

### Configuration

| File | Purpose |
|------|---------|
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |
| `.clang-format` | C/C++ formatting (LLVM, 80 cols) |
| `.style.yapf` | Python formatting (Google, 120 cols) |
| `.isort.cfg` | Python import sorting |
| `.yamllint` | YAML linting rules |

---

## Build System & Toolchains

### Makefile Targets

```bash
make all                # Build toolchain + emulators + docs
make toolchain          # Build LLVM, compiler-rt, picolibc
make emulators          # Build QEMU, Banshee, GVSoC, etc.
make docs               # Build Sphinx documentation
make format             # Format Python and C code
make lint               # Lint without modifying
make clean-docs         # Clean documentation build
```

### CMake Configuration

**Platform selection:**

```bash
cd DeeployTest
cmake -Dplatform=Siracusa -Duse_dma=ON -Dcores=8 ..
make
```

**Platform options:**
- `Generic`: Host CPU
- `MemPool`: MemPool cluster
- `Snitch`: Snitch cluster
- `Siracusa`: Siracusa SoC
- `SoftHier`: Software hierarchy
- `Chimera`: Chimera SoC

**Toolchain options:**
- `GCC`: RISC-V GCC
- `LLVM`: LLVM/Clang

### Environment Variables

```bash
export MINIMALLOC_INSTALL_DIR=${INSTALL_PREFIX}/minimalloc
export PULP_SDK_HOME=${INSTALL_PREFIX}/pulp-sdk
export CHIMERA_SDK_HOME=${INSTALL_PREFIX}/chimera-sdk
export SNITCH_HOME=${INSTALL_PREFIX}/snitch_cluster
export GVSOC_INSTALL_DIR=${INSTALL_PREFIX}/gvsoc
export SOFTHIER_INSTALL_DIR=${INSTALL_PREFIX}/softhier
export LLVM_INSTALL_DIR=${INSTALL_PREFIX}/llvm
export MEMPOOL_HOME=${INSTALL_PREFIX}/mempool
export PATH=${INSTALL_PREFIX}/qemu/bin:${INSTALL_PREFIX}/banshee:$PATH

# Source platform config (for Siracusa)
source ${PULP_SDK_HOME}/configs/siracusa.sh
```

---

## CI/CD Pipeline

### Workflow Organization

```
.github/workflows/
├── ci-deeploy.yml                      # Main CI (lint + unit tests)
├── ci-lint.yml                         # Linting only
├── ci-platform-generic.yml             # Generic CPU tests
├── ci-platform-cortexm.yml             # CortexM tests
├── ci-platform-mempool.yml             # MemPool tests
├── ci-platform-siracusa.yml            # Siracusa basic tests
├── ci-platform-siracusa-tiled.yml      # Siracusa tiling tests
├── ci-platform-siracusa-neureka-tiled.yml  # Siracusa + Neureka
├── ci-platform-snitch.yml              # Snitch basic tests
├── ci-platform-snitch-tiled.yml        # Snitch tiling tests
├── ci-platform-softhier.yml            # SoftHier tests
├── ci-platform-chimera.yml             # Chimera tests
├── docker-build-deeploy.yml            # Docker image build
├── docker-build-toolchain.yml          # Toolchain Docker image
└── infra-generate-documentation.yml    # Docs deployment
```

### CI Stages

1. **Linting** (`ci-lint.yml`)
   - SPDX license headers (reuse)
   - Python formatting (yapf, isort, autoflake)
   - C formatting (clang-format)
   - YAML linting

2. **Unit Tests** (`ci-deeploy.yml`)
   - `pytest testTypes.py`
   - `pytest testTilerExtension.py`
   - `pytest testMemoryLevelExtension.py`

3. **Platform Tests** (`ci-platform-*.yml`)
   - Code generation
   - Compilation with platform toolchain
   - Simulation on platform emulator
   - Output validation

4. **Documentation** (`infra-generate-documentation.yml`)
   - Build Sphinx docs
   - Deploy to GitHub Pages

### Triggering CI

**Automatic triggers:**
- Push to `main` or `devel` branch
- Pull request to `main` or `devel`

**Manual trigger:**
- GitHub Actions UI → "Run workflow"

### CI Artifacts

**Logs:**
- Test outputs
- Compiler warnings/errors
- Simulator logs

**Generated files:**
- C code (TEST_<PLATFORM>/ directories)
- Compiled binaries
- Profiling data

---

## Tips for AI Assistants

### 1. Understanding the Codebase

**Start with these files:**

1. `README.md` - Project overview
2. `Deeploy/DeeployTypes.py` - Core compilation framework
3. `Deeploy/Targets/Generic/Platform.py` - Reference platform
4. `DeeployTest/testUtils/testRunner.py` - Testing framework

**Mental model:**

- **ONNX graph** → (parse) → **OperatorRepresentation** → (bind) → **CodeSnippet** → (generate) → **C code**
- **Extensions** wrap `NetworkDeployer` to add functionality (tiling, memory levels, engines)
- **Platforms** define operator mappings and code templates
- **Tests** validate end-to-end compilation and execution

### 2. Code Navigation

**Finding operator implementations:**

```bash
# Find Conv2D implementation
grep -r "Conv2DParser" Deeploy/Targets/

# Find templates
find Deeploy/Targets/ -name "*ConvTemplate.py"

# Find bindings
grep -r "Conv2DBinding" Deeploy/Targets/
```

**Understanding compilation flow:**

1. Start at `NetworkDeployer.generateFunction()`
2. Trace through `parse()` → `lower()` → `bind()` → `codeTransform()` → `generate()`
3. For specific operator, find its `NodeParser`, `NodeTypeChecker`, `NodeBinding`, `NodeTemplate`

### 3. Common Pitfalls

**Pitfall 1: Modifying wrapped deployer**

When using extensions, always modify the **outer wrapper**, not the inner deployer:

```python
# Wrong
deployer = SiracusaDeployer(...)
deployer = TilerDeployerWrapper(deployer, ...)
deployer.memoryHierarchy = ...  # Modifies TilerDeployerWrapper

# Correct
deployer = SiracusaDeployer(...)
deployer.memoryHierarchy = ...
deployer = TilerDeployerWrapper(deployer, ...)  # Wrapper reads from deployer
```

**Pitfall 2: Type mismatches**

Ensure type consistency across parsing, type checking, and binding:

```python
# Parser extracts input types
parser.parseNodeCtxt(node, ctxt)  # Returns OperatorRepresentation with typed inputs

# TypeChecker validates
checker.typeCheck(operatorRepresentation)  # Checks input.dtype matches expected

# Binding selects template
binding.signatureConstraint = ([int8_t], [int32_t])  # Must match actual types
```

**Pitfall 3: Template variable scoping**

Mako templates have access to `operatorRepresentation` attributes:

```python
# In template
${data_in}      # Access operatorRepresentation.inputs[0]
${kernel_h}     # Access operatorRepresentation.attrs['kernel_h']
${name}         # Access operatorRepresentation.nodeName
```

**Pitfall 4: Buffer lifecycle**

Understand buffer types:

- `ConstantBuffer`: Weights/biases (allocated once, never freed)
- `VariableBuffer`: Inputs/outputs (allocated by caller)
- `TransientBuffer`: Temporaries (allocated/freed per layer)

**Pitfall 5: Extension ordering**

Extensions are applied in reverse order (outermost first):

```python
deployer = NetworkDeployer(...)
deployer = MemoryLevelDeployer(deployer, ...)      # Applied 2nd
deployer = TilerDeployerWrapper(deployer, ...)     # Applied 1st
```

### 4. Debugging Strategies

**Strategy 1: Print intermediate representations**

```python
# After parsing
for node in deployer.graph.node:
    print(f"{node.name}: {node.op_type}")

# After binding
for snippet in deployer.codeSnippets:
    print(f"{snippet.name}:")
    print(f"  Inputs: {snippet.operatorRepresentation.inputs}")
    print(f"  Outputs: {snippet.operatorRepresentation.outputs}")
    print(f"  Attrs: {snippet.operatorRepresentation.attrs}")
```

**Strategy 2: Visualize graph**

```python
from DeeployTest.testUtils.graphDebug import visualizeGraph
visualizeGraph(deployer.graph, filename='graph.png')
```

**Strategy 3: Incremental compilation**

```python
deployer.parse()
print("Parse OK")

deployer.lower()
print("Lower OK")

deployer.typeCheck()
print("TypeCheck OK")

deployer.bind()
print("Bind OK")
```

**Strategy 4: Use DEFAULT_LOGGER**

```python
from Deeploy.Logging import DEFAULT_LOGGER

DEFAULT_LOGGER.info("Starting compilation")
DEFAULT_LOGGER.debug(f"Processing node: {node.name}")
DEFAULT_LOGGER.warning(f"Type mismatch: {input_type} != {expected_type}")
```

### 5. Performance Considerations

**Optimization priorities:**

1. **Correctness first** - Validate output matches ONNX reference
2. **Memory efficiency** - Minimize L1 memory usage (most constrained)
3. **Compute efficiency** - Optimize kernel implementations
4. **Code size** - Minimize flash/ROM footprint

**Profiling workflow:**

1. Enable cycle measurement: `CycleMeasurement()` pass
2. Run on simulator with profiling enabled
3. Analyze hotspots: `profiling2csv.py`
4. Optimize critical kernels

**Memory optimization:**

1. Use tiling for large tensors
2. Reuse transient buffers
3. Minimize constant buffer duplication
4. Consider mixed-precision quantization

### 6. Working with Third-Party Code

**Submodules:**

- Don't modify third-party code directly
- Wrapper/adapter pattern for integration
- Document any patches in `CHANGELOG.md`

**License compatibility:**

- Apache 2.0 (Deeploy)
- MIT (some scripts)
- CC BY 4.0 (docs/tests)
- Check `LICENSES/` for full details

### 7. Communication Best Practices

**When asking for clarification:**

- Reference specific files and line numbers
- Include relevant code snippets
- Describe expected vs. actual behavior
- Provide minimal reproducible example

**When explaining changes:**

- Describe motivation (why, not just what)
- Link to related issues/PRs
- Include before/after comparisons
- Document breaking changes

**When reviewing code:**

- Focus on correctness, performance, maintainability
- Suggest concrete improvements with examples
- Acknowledge good practices
- Be respectful and constructive

---

## Appendix: Quick Reference

### Key Classes Hierarchy

```
BaseType
├── Immediate
│   ├── IntegerImmediate
│   └── FloatImmediate
├── Pointer
└── Struct

VariableBuffer
├── TransientBuffer
├── ConstantBuffer
└── StructBuffer

NodeParser (ABC)
└── [Conv2DParser, MatMulParser, AddParser, ...]

NodeTypeChecker (ABC)
└── [ConvChecker, MatMulChecker, AddChecker, ...]

TopologyOptimizationPass (ABC)
└── [QuantizationPass, FusionPass, ConstantFoldingPass, ...]

CodeTransformationPass (ABC)
└── [MemoryAllocation, Closure, PrintInputs, CycleMeasurement, ...]

NetworkDeployer
└── [GenericDeployer, SiracusaDeployer, SnitchDeployer, ...]
    └── NetworkDeployerWrapper
        ├── SignPropDeployer
        ├── MemoryLevelDeployer
        ├── TilerDeployerWrapper
        └── EngineColoringDeployerWrapper
```

### Common Commands

```bash
# Installation
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com

# Testing
cd DeeployTest
python testRunner_generic.py -t Tests/Adder
python testRunner_siracusa.py -t Tests/testConv --cores=8

# Formatting
make format
make lint

# Building
make toolchain
make emulators
make docs

# Pre-commit
pre-commit install --hook-type pre-push
pre-commit run --all-files
```

### Useful Paths

- **Main source:** `/Deeploy/`
- **Tests:** `/DeeployTest/Tests/`
- **Docs:** `/docs/`
- **CI:** `/.github/workflows/`
- **Libs:** `/TargetLibraries/`
- **Toolchains:** `/toolchain/`
- **Config:** `/pyproject.toml`, `/CMakeLists.txt`, `/Makefile`

### External Resources

- **Documentation:** https://pulp-platform.github.io/Deeploy/
- **Repository:** https://github.com/pulp-platform/Deeploy
- **Issues:** https://github.com/pulp-platform/Deeploy/issues
- **PULP Platform:** https://pulp-platform.org/
- **ONNX:** https://onnx.ai/

---

**End of CLAUDE.md**

*This document is maintained by the Deeploy development team. For updates or corrections, please submit a pull request.*
