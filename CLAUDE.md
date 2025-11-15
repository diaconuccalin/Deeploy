# CLAUDE.md - AI Assistant Guide for Deeploy

This document provides comprehensive guidance for AI assistants working with the Deeploy codebase. It covers the repository structure, development workflows, conventions, and best practices to ensure efficient and consistent contributions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Development Environment Setup](#development-environment-setup)
4. [Architecture and Key Concepts](#architecture-and-key-concepts)
5. [Development Workflow](#development-workflow)
6. [Code Style and Conventions](#code-style-and-conventions)
7. [Testing](#testing)
8. [Build System](#build-system)
9. [Common Tasks](#common-tasks)
10. [Contributing Guidelines](#contributing-guidelines)
11. [Important Files Reference](#important-files-reference)

---

## Project Overview

**Deeploy** is an ONNX-to-C compiler that generates low-level optimized C code for multi-cluster, heterogeneous System-on-Chips (SoCs). It is developed as part of the PULP (Parallel Ultra-Low-Power) project, a collaboration between ETH Zurich and the University of Bologna.

### Key Features
- ONNX model compilation to optimized C code
- Support for multiple heterogeneous platforms (ARM, RISC-V, accelerators)
- Advanced memory tiling and optimization
- DMA scheduling and asynchronous execution support
- Extensive test coverage with simulator integration

### Supported Platforms
1. **Generic CPU** - Host machine execution
2. **ARM Cortex-M** - Microcontroller processors (QEMU simulator)
3. **MemPool + ITA** - RISC-V parallel cluster with accelerator (Banshee simulator)
4. **Siracusa** - PULP-based heterogeneous platform (GVSoC simulator)
5. **Snitch Cluster** - RISC-V cluster (GVSoC simulator)
6. **SoftHier** - Software hierarchy model (GVSoC simulator)
7. **Chimera** - PULP Chimera platform (GVSoC simulator)
8. **Neureka** - Specialized hardware accelerator

---

## Repository Structure

### Top-Level Directory Layout

```
/home/user/Deeploy/
├── Deeploy/                    # Core Python compiler framework (303 .py files)
│   ├── AbstractDataTypes.py    # Base type system abstractions
│   ├── DeeployTypes.py         # Core data structures (126KB)
│   ├── Logging.py              # Unified logging framework
│   ├── CommonExtensions/       # Shared compiler passes and utilities
│   ├── TilingExtension/        # Memory tiling and data movement
│   ├── FutureExtension/        # Asynchronous execution support
│   ├── MemoryLevelExtension/   # Memory hierarchy support
│   ├── EngineExtension/        # Hardware accelerator support
│   └── Targets/                # Platform-specific implementations
│       ├── Generic/            # Baseline CPU implementation
│       ├── CortexM/            # ARM Cortex-M support
│       ├── MemPool/            # MemPool cluster
│       ├── Snitch/             # Snitch cluster
│       ├── PULPOpen/           # PULP platform
│       ├── Neureka/            # Neureka accelerator
│       ├── Chimera/            # Chimera platform
│       └── SoftHier/           # Software hierarchy
│
├── DeeployTest/                # Testing framework and test cases
│   ├── testRunner_*.py         # Platform-specific test runners (25 files)
│   ├── test*.py                # Unit tests (14 files)
│   ├── Tests/                  # Test cases (94+ directories)
│   │   ├── Adder/              # Example: simple addition test
│   │   │   ├── network.onnx    # ONNX model
│   │   │   ├── inputs.npz      # NumPy test inputs
│   │   │   └── outputs.npz     # Expected outputs
│   │   ├── MobileNetv2/        # Neural network tests
│   │   ├── microLlama/         # Language model tests
│   │   └── ...
│   └── Platforms/              # Platform-specific test outputs
│       ├── Generic/
│       ├── MemPool/
│       ├── Snitch/
│       └── ...
│
├── TargetLibraries/            # Platform-specific C/C++ microkernels (154 C files)
│   ├── Generic/                # Reference implementation (29 files)
│   │   ├── inc/kernel/         # Header files (GEMM, Conv, etc.)
│   │   └── src/                # Source implementations
│   ├── MemPool/                # MemPool optimized kernels (15 files)
│   ├── Snitch/                 # Snitch optimized kernels (15 files)
│   ├── PULPOpen/               # PULP optimizations (16 files)
│   └── CMSIS/                  # ARM CMSIS-NN (submodule)
│
├── cmake/                      # CMake toolchain and configuration files
│   ├── Util.cmake              # Common utility functions
│   ├── common.cmake            # Shared configuration
│   ├── simulation.cmake        # Simulator setup
│   ├── mempool/                # MemPool configurations
│   ├── snitch/                 # Snitch configurations
│   ├── pulp/                   # PULP configurations
│   ├── cmsis/                  # ARM configurations
│   └── generic/                # Generic CPU configurations
│
├── docs/                       # Documentation (Sphinx)
├── scripts/                    # Utility scripts
│   ├── run_clang_format.py     # C code formatting
│   ├── gen_changelog.py        # Changelog generation
│   └── reuse_skip_wrapper.py   # License compliance
│
├── Container/                  # Docker build configuration
├── toolchain/                  # External toolchain installation scripts
├── .github/workflows/          # GitHub CI/CD workflows
├── LICENSES/                   # License files
├── CMakeLists.txt              # Root CMake configuration
├── Makefile                    # Toolchain installation targets
├── pyproject.toml              # Python project metadata
├── setup.py                    # Python package setup
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Release history
└── CLAUDE.md                   # This file
```

---

## Development Environment Setup

### Installation Methods

#### Option 1: Using Docker (Recommended for Quick Start)
```bash
# Pull the official Docker image
docker pull ghcr.io/pulp-platform/deeploy:main

# Create and start container with workspace mounted
docker run -it --name deeploy_main -v $(pwd):/app/Deeploy ghcr.io/pulp-platform/deeploy:main

# Inside the container, install Deeploy in editable mode
cd Deeploy
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com
```

#### Option 2: Local Installation
```bash
# Clone repository with submodules
git clone https://github.com/pulp-platform/Deeploy.git
cd Deeploy
git submodule update --init --recursive

# Install Deeploy
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com

# Install development dependencies
pip install -r requirements-dev.txt

# (Optional) Install pre-commit hooks
pip install pre-commit
pre-commit install --hook-type pre-push
```

### Python Dependencies
Core dependencies (from `pyproject.toml`):
- **Python**: >=3.10
- **ONNX ecosystem**: onnx, onnxruntime, onnx-graphsurgeon==0.3.20
- **Core libs**: numpy<2.0.0, protobuf==4.23.3
- **Code generation**: mako (template engine)
- **Optimization**: ortools (constraint solving)
- **Utilities**: toml, argparse, pytest, plotly, coloredlogs

Development dependencies (from `requirements-dev.txt`):
- Code quality: yapf, isort, autoflake, yamllint
- Documentation: sphinx, sphinx-rtd-theme, sphinx-copybutton
- Testing: pytest, pytest-xdist
- License compliance: reuse

---

## Architecture and Key Concepts

### Compilation Flow

```
ONNX Model
    ↓
Parser (per Target)
    ↓
Type Checking (TypeChecker.py)
    ↓
Network Deployer (NetworkDeployers/)
    ↓
Topology Optimization Passes
    ↓
[Optional] Tiling Extension
    → Memory Scheduler
    → Memory Constraints
    → DMA Scheduling
    ↓
Bindings (node-to-template mapping)
    ↓
Code Transformation Passes
    ↓
Code Generation (Mako Templates)
    ↓
C Code Output
    ↓
CMake/Build System → Binary
    ↓
Simulator Execution
```

### Key Python Modules

#### Core Abstraction Layer
- **`DeeployTypes.py`** (126KB): Central data structures
  - `CodeSnippet`, `CodeGenVerbosity` - Code generation control
  - `OperatorRepresentation` - Node template representation
  - `DeploymentPlatform` - Hardware platform abstraction
  - `TopologyOptimizer` - Graph optimization framework
  - ONNX integration and graph manipulation

- **`AbstractDataTypes.py`**: Type system foundation
  - `BaseType`, `Pointer`, `Struct`, `VoidType` classes
  - Type variables and generic type definitions
  - Property descriptors for class attributes

- **`Logging.py`**: Centralized logging
  - `DEFAULT_LOGGER` - Use this instead of `print()` statements
  - ANSI color codes for structured output
  - Success/failure markers

#### Extension Modules

1. **CommonExtensions/** - Shared compiler infrastructure
   - `DataTypes.py` - Standard data types (int8_t through int64_t, uint*, float16/32/64, bfloat16)
   - `CodeTransformationPasses/` - Code generation passes:
     - `Closure.py` - Generate function closures
     - `MemoryAllocation.py` - Memory allocation optimization
     - `CycleMeasurement.py` - Performance cycle counting
     - `PrintInputs.py` - Debug I/O functionality
   - `OptimizationPasses/TopologyOptimizationPasses/` - Graph rewrites
   - `OptimizationPasses/BindingsOptimizationPasses/` - Node binding optimization
   - `NetworkDeployers/` - Deployment strategies
   - `TypeCheckers/` - Type validation (`SignPropTypeChecker.py`)

2. **TilingExtension/** - Memory tiling system
   - **Purpose**: Enable execution of large models on memory-constrained devices
   - `TilerExtension.py` - Main tiling framework
   - `TilingCodegen.py` - Code generation for tiled operations
   - `MemoryScheduler.py` - Memory scheduling algorithms
   - `MemoryConstraints.py` - Memory constraint definitions
   - `TileConstraint.py` - Tile size constraints
   - `AsyncDma.py` - Asynchronous DMA abstractions
   - `CodeTransformationPasses/` - Single/double buffering code generation

3. **FutureExtension/** - Asynchronous execution
   - Future/Promise abstractions for async operations
   - `Future.py`, `Bindings/`, `CodeTransformationPasses/`

4. **MemoryLevelExtension/** - Memory hierarchy
   - L1, L2, L3 memory level definitions
   - Memory level annotation passes

5. **EngineExtension/** - Hardware accelerator support
   - Accelerator scheduling ("engine coloring")
   - Engine-aware optimization passes

#### Target Platform Structure

Each target in `Targets/` follows this standard structure:

```
Targets/<PlatformName>/
├── Platform.py                 # Hardware capabilities and specs
├── Deployer.py                 # Orchestrates code generation
├── Bindings.py                 # Maps ONNX ops to templates
├── DataTypes.py                # Platform-specific data types
├── Layers.py                   # Layer implementations
├── Parsers.py                  # ONNX parser implementations
├── TypeCheckers.py             # Type validation rules
├── Tiler.py                    # Tiling implementation (if applicable)
├── Templates/                  # Mako/Jinja2 code generation templates
├── TileConstraints/            # Memory and dimension constraints
├── TopologyOptimizationPasses/ # Platform-specific graph optimizations
└── CodeTransformationPasses/   # DMA, synchronization, etc.
```

**Example: Generic Target Deployer Pattern**
From `Deeploy/Targets/Generic/Deployer.py`:
```python
class GenericDeployer(SignPropDeployer):
    def __init__(self, graph, deploymentPlatform, inputTypes,
                 loweringOptimizer, scheduler, name, ...):
        super().__init__(...)
        # Add platform-specific optimization passes
        self.loweringOptimizer.passes += [
            TransposeMatmulInputsPass(),
            NCHWtoNHWCPass(self.default_channels_first),
            TransposeMergePass(),
            TransposeConstOptPass(),
            DebugPrintMergePass()
        ]
```

### C Kernel Library Organization

**Generic Library** (`TargetLibraries/Generic/`) - Reference implementation:
- **Headers** (`inc/kernel/`):
  - Math operations: `Gemm.h`, `MatMul.h`, `Convolution.h`, `DWConvolution.h`
  - Activations: `GELU.h`, `Hardswish.h`, `Relu.h`, `Softmax.h`
  - Normalization: `Layernorm.h`, `RMSNorm.h`, `BatchNorm.h`
  - Utilities: `MaxPool.h`, `RequantShift.h`, `Div.h`

- **Sources** (`src/`): Type-specific implementations
  - `Gemm_s8.c`, `Gemm_fp32.c` - 8-bit integer and 32-bit float variants
  - `MatMul_s8.c`, `MatMul_s16.c`, `MatMul_s32.c`, `MatMul_fp32.c`
  - `Convolution_s8.c`, `Convolution_fp32.c`
  - Pattern: `<Operation>_<DataType>.c`

**Platform-Specific Libraries** extend or optimize the generic implementation:
- **MemPool**: ITA (Inter-Task Accelerator) support, multi-core optimizations
- **Snitch**: Cluster synchronization, DMA optimizations
- **PULPOpen**: Advanced PULP-specific kernels, pulp-nn integration

---

## Development Workflow

### Branch Strategy
- **Main branch**: Stable releases
- **Devel branch**: Active development
- **Feature branches**: All new work must be done on feature branches
- **Pull requests**: Always target the `devel` branch

### Creating a Pull Request

1. **Create feature branch**:
   ```bash
   git checkout devel
   git pull origin devel
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit**:
   ```bash
   # Make your changes
   git add <files>
   git commit -m "Brief description of changes"
   ```

3. **Update CHANGELOG.md**:
   - Add entry under "Unreleased" section
   - Include PR title and link in "List of Pull Requests"
   - Add concise description under appropriate category (Added/Changed/Fixed/Removed)
   - See `CHANGELOG.md` lines 1-100 for format examples

4. **Format code**:
   ```bash
   make format  # Or just: make lint (to check without modifying)
   ```

5. **Run tests**:
   ```bash
   cd DeeployTest
   python testRunner_generic.py -t Tests/Adder  # Basic smoke test
   # Run platform-specific tests as needed
   ```

6. **Push and create PR**:
   ```bash
   git push -u origin feature/your-feature-name
   # Create PR on GitHub targeting 'devel' branch
   ```

### Git Commit Message Guidelines
- Use imperative mood: "Add feature" not "Added feature"
- Be concise but descriptive
- Reference issue numbers when applicable: "Fix #123: Description"
- Follow existing commit message patterns in `git log`

---

## Code Style and Conventions

### Python Code Style

**Formatter**: `yapf` (based on Google style)
- **Line length**: 120 characters
- **Configuration**: `.style.yapf`
- **Import sorting**: `isort` with configuration in `.isort.cfg`
  - Line length: 120
  - Multi-line output mode: 2
  - No trailing commas in imports

**Key conventions**:
- Use `DEFAULT_LOGGER` from `Deeploy.Logging` instead of `print()` statements
- Class names: `PascalCase` (e.g., `GenericDeployer`, `SignPropTypeChecker`)
- Function names: `camelCase` (e.g., `hoistReference`, `serializeTilingSolution`)
- Private methods: prefix with `_` (e.g., `_mangleNodeNames`, `_permuteList`)
- Constants: `UPPER_CASE` (e.g., `LLVM_INSTALL_DIR`, `DEFAULT_LOGGER`)

**File headers**: All files must have SPDX license headers
```python
# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
```

**Running formatters**:
```bash
# Format all Python files
yapf -ri -e "*/TEST_*/" -e "*/third_party/" -e "install/" -e "toolchain/" .

# Sort imports
isort .

# Both at once
make format
```

### C Code Style

**Formatter**: `clang-format` (based on LLVM style)
- **Line length**: 80 characters
- **Configuration**: `.clang-format`
- **Standard**: C99

**Key conventions**:
- Function names: `snake_case` or `camelCase` (consistent within file)
- Type names: `typedef` with `_t` suffix (e.g., `int8_t`, `uint32_t`)
- Macros: `UPPER_CASE`
- Include guards: `__<FILENAME>_H__` format

**File headers**: Same SPDX format as Python
```c
// SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0
```

**Running formatter**:
```bash
# Format all C files
python scripts/run_clang_format.py

# Or via make
make format
```

### YAML Code Style
- **Linter**: `yamllint`
- **Configuration**: `.yamllint`
- Run with: `yamllint .`

### License Compliance
- All files must have SPDX license headers
- Use `reuse` tool to check compliance:
  ```bash
  reuse lint  # Check all files
  reuse annotate --license Apache-2.0 --copyright "ETH Zurich and University of Bologna" <file>
  ```

---

## Testing

### Test Organization

#### Test Structure
Each test case in `DeeployTest/Tests/` contains:
```
Tests/<TestName>/
├── network.onnx       # ONNX model file
├── inputs.npz         # NumPy input tensors
├── outputs.npz        # Expected output tensors
└── activations.npz    # (Optional) Intermediate activations
```

#### Test Runners
Platform-specific test runners in `DeeployTest/`:
- `testRunner_generic.py` - Generic CPU (no tiling)
- `testRunner_cortexm.py` - ARM Cortex-M (QEMU)
- `testRunner_mempool.py` - MemPool cluster (Banshee)
- `testRunner_snitch.py` - Snitch cluster (GVSoC)
- `testRunner_snitch_dma.py` - Snitch with async DMA
- `testRunner_siracusa.py` - Siracusa platform
- `testRunner_tiled_siracusa.py` - Siracusa with tiling
- `testRunner_tiled_siracusa_w_neureka.py` - Siracusa with Neureka accelerator
- `testRunner_tiled_snitch.py` - Snitch with tiling
- `testRunner_softhier.py` - SoftHier platform
- `testRunner_chimera.py` - Chimera platform

**Test runner pattern**:
```python
from testUtils.testRunner import TestRunner, TestRunnerArgumentParser

parser = TestRunnerArgumentParser(
    tiling_arguments=False,  # or True for tiling support
    description="Description of platform and test configuration"
)
args = parser.parse_args()

testRunner = TestRunner(
    platform="Generic",  # or other platform name
    simulator="host",    # or "qemu", "banshee", "gvsoc"
    tiling=False,        # or True
    argument_parser=parser
)

testRunner.run()
```

### Running Tests

#### Basic Test Execution
```bash
cd DeeployTest

# Run simple test on Generic platform
python testRunner_generic.py -t Tests/Adder

# Run on specific platform
python testRunner_cortexm.py -t Tests/Adder
python testRunner_mempool.py -t Tests/Adder
python testRunner_snitch.py -t Tests/Adder

# Run with tiling
python testRunner_tiled_siracusa.py -t Tests/testMatMul --cores=8 --l1=16000
```

#### Test Output Locations
Generated code is placed in platform-specific directories:
- `DeeployTest/TEST_GENERIC/` - Generic platform outputs
- `DeeployTest/TEST_MEMPOOL/` - MemPool outputs
- `DeeployTest/TEST_SNITCH/` - Snitch outputs
- etc.

Each contains:
```
TEST_<PLATFORM>/Tests/<TestName>/
├── inc/              # Generated headers
├── src/              # Generated C source
├── Network.c         # Main network implementation
├── main.c            # Entry point
└── CMakeLists.txt    # Build configuration
```

#### Unit Tests
Python unit tests are in `DeeployTest/test*.py`:
```bash
pytest DeeployTest/testTypes.py
pytest DeeployTest/testTilerExtension.py
pytest DeeployTest/testMemoryLevelExtension.py
```

### CI/CD Testing

GitHub Actions workflows in `.github/workflows/`:
- `ci-lint.yml` - Code formatting and license checks
- `ci-platform-generic.yml` - Generic platform tests
- `ci-platform-cortexm.yml` - Cortex-M tests
- `ci-platform-mempool.yml` - MemPool tests
- `ci-platform-siracusa.yml` - Siracusa tests
- `ci-platform-snitch.yml` - Snitch tests
- etc.

All workflows:
1. Build Deeploy in Docker container
2. Install dependencies
3. Run platform-specific test suite
4. Report results

---

## Build System

### CMake Structure

#### Root Configuration
`CMakeLists.txt` (top-level):
- Platform selection via `PLATFORM` variable
- Compiler selection: LLVM or GCC
- Output directories: `lib/`, `bin/`
- C standard: C99
- ccache integration for faster builds

#### Platform-Specific CMake Files
Located in `cmake/<platform>/`:

**MemPool** (`cmake/mempool/`):
- `mempool.cmake` - Standard MemPool configuration
- `mempool_ita.cmake` - ITA accelerator variant
- `minpool.cmake` - Minimal MemPool
- `toolchain_gcc.cmake`, `toolchain_llvm.cmake` - Compiler toolchains

**Snitch** (`cmake/snitch/`):
- `snitch.cmake` - Main configuration
- `toolchain_llvm.cmake` - LLVM toolchain

**PULP** (`cmake/pulp/`):
- `pulp.cmake` - PULP-Open configuration
- `toolchain_gcc.cmake`, `toolchain_llvm.cmake`

**ARM** (`cmake/cmsis/`):
- `cmsis.cmake` - ARM Cortex-M configuration
- `qemu.cmake` - QEMU integration
- `toolchain_gcc.cmake`, `toolchain_llvm.cmake`

**Generic** (`cmake/generic/`):
- `generic.cmake` - Host CPU configuration
- `toolchain_llvm.cmake`

#### CMake Utility Files
- `cmake/Util.cmake` - Common utility functions
- `cmake/common.cmake` - Shared configuration
- `cmake/simulation.cmake` - Simulator setup

### Makefile (Toolchain Installation)

The root `Makefile` handles installation of toolchains and simulators:

**Toolchain targets**:
```bash
make toolchain              # Install all toolchain components
make llvm                   # LLVM/Clang compiler
make llvm-compiler-rt-riscv # RISC-V compiler runtime
make llvm-compiler-rt-arm   # ARM compiler runtime
make picolibc-arm           # Picolibc for ARM
make picolibc-riscv         # Picolibc for RISC-V
```

**Emulator/simulator targets**:
```bash
make emulators     # Install all simulators
make qemu          # QEMU for ARM
make banshee       # Banshee for MemPool
make mempool       # MemPool simulator
make snitch_runtime # Snitch cluster
make pulp-sdk      # PULP SDK (includes GVSoC)
make chimera-sdk   # Chimera SDK
```

**Installation directories** (configurable via environment variables):
- `INSTALL_PREFIX` - Base installation directory (default: `install/`)
- `LLVM_INSTALL_DIR` - LLVM installation (default: `install/llvm/`)
- `PULP_SDK_INSTALL_DIR` - PULP SDK (default: `install/pulp-sdk/`)
- `SNITCH_INSTALL_DIR` - Snitch (default: `install/snitch_cluster/`)
- `QEMU_INSTALL_DIR` - QEMU (default: `install/qemu/`)
- etc.

**Commit hashes** for dependency versions are pinned in the Makefile:
- `LLVM_COMMIT_HASH`
- `PULP_SDK_COMMIT_HASH`
- `BANSHEE_COMMIT_HASH`
- `MEMPOOL_COMMIT_HASH`
- `SNITCH_COMMIT_HASH`
- `GVSOC_COMMIT_HASH`
- etc.

**Required environment variables** after installation:
```bash
export MINIMALLOC_INSTALL_DIR=${INSTALL_DIR}/minimalloc
export PULP_SDK_HOME=${INSTALL_DIR}/pulp-sdk
export CHIMERA_SDK_HOME=${INSTALL_DIR}/chimera-sdk
export SNITCH_HOME=${INSTALL_DIR}/snitch_cluster
export GVSOC_INSTALL_DIR=${INSTALL_DIR}/gvsoc
export SOFTHIER_INSTALL_DIR=${INSTALL_DIR}/softhier
export LLVM_INSTALL_DIR=${INSTALL_DIR}/llvm
export MEMPOOL_HOME=${INSTALL_DIR}/mempool
export CMAKE=$(which cmake)
export PATH=${QEMU_INSTALL_DIR}/bin:${BANSHEE_INSTALL_DIR}:$PATH
export PATH=~/.cargo/bin:$PATH

# Also source PULP SDK configuration
source ${PULP_SDK_HOME}/configs/siracusa.sh
```

### Building Generated Code

After running a test runner, build the generated C code:

```bash
cd DeeployTest/TEST_GENERIC/Tests/Adder
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./main  # Run the executable
```

---

## Common Tasks

### Adding Support for a New ONNX Operator

1. **Create parser** in `Targets/<Platform>/Parsers.py`:
   ```python
   class MyOpParser(Parser):
       def parse(self, node: gs.Node, ...) -> OperatorRepresentation:
           # Parse ONNX node attributes
           # Return OperatorRepresentation
   ```

2. **Create C kernel** in `TargetLibraries/<Platform>/`:
   - Header: `inc/kernel/MyOp.h`
   - Source: `src/MyOp_s8.c`, `src/MyOp_fp32.c`, etc.

3. **Create template** in `Targets/<Platform>/Templates/`:
   - Mako template for code generation
   - Example: `MyOpTemplate.py`

4. **Create binding** in `Targets/<Platform>/Bindings.py`:
   ```python
   class MyOpBinding(NodeBinding):
       def __init__(self):
           super().__init__(
               operatorRepresentation=MyOpParser.operatorRepresentation,
               templateGenerator=MyOpTemplate
           )
   ```

5. **Add to platform** in `Targets/<Platform>/Platform.py`:
   - Register parser
   - Register binding
   - Add to type matchers if needed

6. **Create test** in `DeeployTest/Tests/testMyOp/`:
   - Generate `network.onnx` with the operator
   - Create `inputs.npz` and `outputs.npz`
   - Run test: `python testRunner_generic.py -t Tests/testMyOp`

7. **Add tile constraints** (if tiling is needed):
   - Create constraint in `Targets/<Platform>/TileConstraints/`

### Adding a New Platform Target

1. **Create directory structure**:
   ```bash
   mkdir -p Deeploy/Targets/NewPlatform/{Templates,TileConstraints,TopologyOptimizationPasses,CodeTransformationPasses}
   ```

2. **Create core files**:
   - `Platform.py` - Define hardware capabilities
   - `Deployer.py` - Extend `SignPropDeployer` or appropriate base
   - `Bindings.py` - Map ONNX ops to templates
   - `DataTypes.py` - Define platform data types
   - `Parsers.py` - ONNX parsers for supported ops
   - `TypeCheckers.py` - Type validation

3. **Create C library**:
   ```bash
   mkdir -p TargetLibraries/NewPlatform/{inc/kernel,src}
   ```
   - Implement kernels for each operator
   - Create `CMakeLists.txt` for library build

4. **Create CMake support**:
   - `cmake/newplatform/newplatform.cmake` - Platform configuration
   - `cmake/newplatform/toolchain_gcc.cmake` or `toolchain_llvm.cmake`

5. **Create test runner**:
   - `DeeployTest/testRunner_newplatform.py`
   - Follow pattern from existing runners

6. **Add CI workflow**:
   - `.github/workflows/ci-platform-newplatform.yml`
   - Follow pattern from existing platform workflows

### Debugging Generated Code

1. **Enable debug outputs** in test runner:
   ```python
   # Add to test runner
   from Deeploy.DeeployTypes import CodeGenVerbosity
   # Set verbosity level when creating deployer
   ```

2. **Use debug transformation passes**:
   ```python
   from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.DebugPasses import DebugPrintMergePass
   # Add to optimizer passes
   ```

3. **Check generated code** in `DeeployTest/TEST_<PLATFORM>/Tests/<TestName>/`:
   - `Network.c` - Main network implementation
   - `inc/` - Generated headers
   - `src/` - Generated sources

4. **Use logging**:
   ```python
   from Deeploy.Logging import DEFAULT_LOGGER
   DEFAULT_LOGGER.info("Debug message")
   DEFAULT_LOGGER.debug("Detailed debug info")
   ```

5. **Visualize ONNX graph**:
   - Use [Netron](https://netron.app/) to visualize `network.onnx` files
   - Inspect node operations and connections

### Working with Memory Tiling

**Purpose**: Execute large models on memory-constrained devices by breaking operations into smaller tiles.

**Key concepts**:
- **Tile**: A subset of tensor data that fits in available memory
- **Memory levels**: L1 (fast, small), L2 (medium), L3 (slow, large)
- **DMA**: Direct Memory Access for efficient data movement
- **Buffering**: Single-buffering (simple) vs double-buffering (overlap compute/transfer)

**Example tiled execution**:
```bash
# Run with explicit memory constraints
python testRunner_tiled_siracusa.py -t Tests/testMatMul --cores=8 --l1=16000

# Check generated tiling code
cat DeeployTest/TEST_SIRACUSA/Tests/testMatMul/Network.c
```

**Adding tile constraints** for a new operator:
1. Create constraint class in `Targets/<Platform>/TileConstraints/`
2. Inherit from `TileConstraint` base class
3. Define `serializeTilingSolution()` method
4. Register constraint in platform's tiler

---

## Contributing Guidelines

### Before Submitting a Pull Request

1. **Ensure code is formatted**:
   ```bash
   make format  # Format all code
   make lint    # Check formatting without changes
   ```

2. **Update CHANGELOG.md**:
   - Add entry under "Unreleased" section
   - Include PR title and number in "List of Pull Requests"
   - Add description under appropriate category (Added/Changed/Fixed/Removed)

3. **Run tests**:
   - At minimum, run basic smoke test: `python testRunner_generic.py -t Tests/Adder`
   - Run platform-specific tests if changes affect specific platforms
   - Add new tests for new features

4. **Check license headers**:
   ```bash
   reuse lint  # Check all files have proper SPDX headers
   ```

5. **Add documentation**:
   - Update docstrings for new functions/classes
   - Update this file (CLAUDE.md) if adding new patterns or conventions
   - Update README.md if adding user-facing features

### Pull Request Requirements

From `CONTRIBUTING.md`:

- **Target branch**: Always create PRs against `devel` branch
- **Early drafts encouraged**: Submit draft PRs to keep development transparent
- **Label draft PRs**: Prefix with "DRAFT:" in title
- **Label refactoring**: Prefix with "REFACTOR:" in title
- **Reference issues**: If PR addresses a specific issue, reference it in description
- **Add tests**: Include at least a proof-of-concept test for new features
- **Regression tests**: If fixing a bug, add a test for the error condition
- **Changelog required**: All PRs must update CHANGELOG.md
- **Licensing**: All contributions accepted under Apache 2.0 license only

### Code Review Process

- **Discussion encouraged**: Comment on open PRs with productive feedback
- **Actionable feedback**: Include clear, actionable items for improvement
- **Compatibility**: Ensure feature ideas are compatible with Deeploy framework
- **Quality standard**: Maintain consistent minimal quality standard
- **Rebase required**: You may be asked to rebase your work against devel branch

### Pre-commit Hooks (Optional but Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks (runs on pre-push)
pre-commit install --hook-type pre-push

# The hooks will automatically run before each push
# To uninstall: pre-commit uninstall
```

---

## Important Files Reference

### Configuration Files

| File | Purpose | Key Settings |
|------|---------|--------------|
| `pyproject.toml` | Python package metadata | Dependencies, version, project info |
| `setup.py` | Python package setup | Minimal setup script |
| `requirements-dev.txt` | Development dependencies | Testing, linting, docs tools |
| `.style.yapf` | Python formatting | Line length: 120, Google style |
| `.isort.cfg` | Import sorting | Line length: 120, multi-line mode: 2 |
| `.clang-format` | C/C++ formatting | Line length: 80, LLVM style |
| `.yamllint` | YAML linting | YAML style rules |
| `.pre-commit-config.yaml` | Pre-commit hooks | yapf, isort, autoflake, yamllint, clang-format, reuse |
| `REUSE.toml` | License compliance | SPDX configuration |
| `.gitignore` | Git exclusions | Ignore build artifacts, temp files |
| `.gitmodules` | Git submodules | CMSIS-NN, pulp-nn-mixed, pulp-nnx |

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, installation, quick start |
| `CONTRIBUTING.md` | Contribution workflow, style guide, PR process |
| `CHANGELOG.md` | Release history, unreleased changes |
| `CONTRIBUTORS.md` | List of contributors |
| `CODEOWNERS` | Code ownership mapping for GitHub |
| `CLAUDE.md` | This file - AI assistant guide |

### Build Files

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Root CMake configuration |
| `Makefile` | Toolchain and simulator installation |
| `cmake/*.cmake` | Platform-specific CMake configurations |
| `TargetLibraries/*/CMakeLists.txt` | Library build configurations |
| `DeeployTest/CMakeLists.txt` | Test build configuration |

### Key Python Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `Deeploy/DeeployTypes.py` | Core data structures | `CodeSnippet`, `OperatorRepresentation`, `DeploymentPlatform` |
| `Deeploy/AbstractDataTypes.py` | Type system | `BaseType`, `Pointer`, `Struct` |
| `Deeploy/Logging.py` | Logging framework | `DEFAULT_LOGGER` |
| `Deeploy/Targets/*/Deployer.py` | Platform deployers | Platform-specific `Deployer` classes |
| `Deeploy/Targets/*/Platform.py` | Hardware definitions | Platform capabilities |
| `Deeploy/Targets/*/Bindings.py` | Op-to-template mapping | `NodeBinding` subclasses |
| `Deeploy/TilingExtension/TilerExtension.py` | Tiling framework | Tiling orchestration |

---

## Quick Reference Commands

### Development
```bash
# Install Deeploy (editable mode)
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com

# Install dev dependencies
pip install -r requirements-dev.txt

# Format code
make format

# Check formatting
make lint

# Check licenses
reuse lint
```

### Testing
```bash
cd DeeployTest

# Run basic test
python testRunner_generic.py -t Tests/Adder

# Run platform-specific tests
python testRunner_cortexm.py -t Tests/Adder
python testRunner_mempool.py -t Tests/Adder
python testRunner_snitch.py -t Tests/Adder

# Run with tiling
python testRunner_tiled_siracusa.py -t Tests/testMatMul --cores=8 --l1=16000

# Run unit tests
pytest DeeployTest/testTypes.py
```

### Build
```bash
# Install toolchains
make toolchain

# Install simulators
make emulators

# Build generated code
cd DeeployTest/TEST_GENERIC/Tests/Adder
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Git Workflow
```bash
# Create feature branch from devel
git checkout devel
git pull origin devel
git checkout -b feature/new-feature

# Make changes, commit, push
git add <files>
git commit -m "Description"
git push -u origin feature/new-feature

# Create PR on GitHub targeting 'devel' branch
```

---

## Additional Resources

- **Documentation**: [Main Branch](https://pulp-platform.github.io/Deeploy/) | [Devel Branch](https://pulp-platform.github.io/Deeploy/branch/devel/)
- **DeepWiki**: [Generated docs](https://deepwiki.com/pulp-platform/Deeploy)
- **GitHub Issues**: [Issue tracker](https://github.com/pulp-platform/Deeploy/issues)
- **Publications**: See README.md for citation information
- **PULP Platform**: [PULP project website](https://pulp-platform.org/)

---

## Notes for AI Assistants

### When Working on This Codebase

1. **Always check file headers**: Ensure SPDX license headers are present
2. **Use logging, not print**: Import and use `DEFAULT_LOGGER` from `Deeploy.Logging`
3. **Follow naming conventions**:
   - Classes: `PascalCase`
   - Functions: `camelCase`
   - Private methods: `_camelCase`
   - Constants: `UPPER_CASE`
4. **Update CHANGELOG.md**: Every PR must include a changelog entry
5. **Format before committing**: Run `make format` before committing
6. **Target devel branch**: All PRs should target the `devel` branch, not `main`
7. **Add tests**: New features require at least a proof-of-concept test
8. **Read existing code first**: Before adding new functionality, check if similar code exists
9. **Preserve patterns**: Follow established patterns in the codebase
10. **Document complex logic**: Add docstrings and comments for non-obvious code

### Common Pitfalls to Avoid

- Don't use `print()` - use `DEFAULT_LOGGER` instead
- Don't skip license headers - all files need SPDX headers
- Don't forget CHANGELOG.md - it's required for all PRs
- Don't create PRs against `main` - use `devel` branch
- Don't add untested code - include tests with new features
- Don't ignore formatting - CI will fail on formatting errors
- Don't hardcode paths - use configuration and environment variables
- Don't break existing tests - run tests before submitting PR

### Understanding the Architecture

The key insight is that Deeploy is a **modular compiler** with:
- **Platform abstraction**: Each target platform is self-contained
- **Extension system**: Tiling, futures, memory levels, engines are optional extensions
- **Pass-based optimization**: Graph transformations via optimization passes
- **Template-based codegen**: Mako templates generate C code
- **Type-driven compilation**: Type checking drives code generation decisions

When adding features, consider:
- Which platform(s) should support it?
- Does it need a new optimization pass?
- Should it be an extension or core functionality?
- What are the type system implications?
- How does it affect memory management?

---

**Last Updated**: 2025-11-15
**Deeploy Version**: 0.2.1 (unreleased)
**Maintained by**: PULP Platform (ETH Zurich & University of Bologna)
