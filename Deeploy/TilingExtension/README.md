# TilingExtension - Memory Tiling for Neural Network Deployment

This directory contains Deeploy's tiling infrastructure for partitioning neural network operations into memory-efficient tiles.

## Quick Links

- 📖 **[CLAUDE.md](../../CLAUDE.md)**: Comprehensive guide to understanding tiling constraints
- 🐛 **[DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md)**: How to debug and visualize tiling solutions
- 💻 **[Example Script](./examples/debug_tiling_example.py)**: Hands-on examples of debugging tools

## Overview

### What is Tiling?

Tiling partitions large neural network operations into smaller chunks that fit in fast on-chip memory:

```
Full Tensor (doesn't fit in L1)     Tiled Approach (fits in L1)
┌─────────────────────────────┐     ┌──────┬──────┬──────┬──────┐
│                             │     │ T0   │ T1   │ T2   │ T3   │
│                             │     ├──────┼──────┼──────┼──────┤
│    32 x 32 x 64 tensor      │ →   │ T4   │ T5   │ T6   │ T7   │
│                             │     ├──────┼──────┼──────┼──────┤
│                             │     │ T8   │ T9   │ T10  │ T11  │
└─────────────────────────────┘     └──────┴──────┴──────┴──────┘
                                    Each tile: 8x8x64 (fits in L1)
```

### Key Components

| File | Purpose |
|------|---------|
| `TileConstraint.py` | Base class for defining tiling constraints |
| `TilerModel.py` | OR-Tools constraint solver for finding valid tilings |
| `TilingCodegen.py` | Code generation from tiling solutions |
| `MemoryConstraints.py` | Memory allocation and scheduling |
| `MemoryScheduler.py` | DMA scheduling and optimization |
| **`TileConstraintDebugger.py`** | **NEW: Debugging and analysis tools** |
| **`TileConstraintVisualizer.py`** | **NEW: Interactive HTML visualizations** |
| **`DEBUGGING_GUIDE.md`** | **NEW: Comprehensive debugging guide** |

## Getting Started

### 1. Understanding Tiling Constraints

Read **[CLAUDE.md](../../CLAUDE.md)** for a comprehensive tutorial on:
- How tiling works
- How to write custom TileConstraints
- Common patterns and examples
- Target-specific constraints

### 2. Using the Debugging Tools

#### Quick Start: Analyze a Constraint Model

```python
from Deeploy.TilingExtension.TileConstraintDebugger import TileConstraintDebugger

# Create debugger
debugger = TileConstraintDebugger(verbose=True)

# Analyze your tiling model
analysis = debugger.analyze_constraint_model(
    tilerModel=myTilerModel,
    operatorName="Conv2D_layer1"
)

# Check for issues
if analysis['warnings']:
    print("⚠️ Warnings found:")
    for warning in analysis['warnings']:
        print(f"  {warning}")
```

#### Quick Start: Visualize a Tiling Solution

```python
from Deeploy.TilingExtension.TileConstraintVisualizer import TileConstraintVisualizer

# Create visualizer
visualizer = TileConstraintVisualizer(output_dir="./my_viz")

# Generate interactive HTML
html_path = visualizer.visualize_tiling_schedule(
    tilingSchedule=schedule,
    variableReplacement=var_replacement,
    operatorName="Conv2D_layer1",
    tensorShapes={
        'data_in': (1, 32, 32, 64),
        'data_out': (1, 30, 30, 128)
    }
)

print(f"Open {html_path} in your browser!")
```

### 3. Run the Examples

```bash
cd Deeploy/TilingExtension/examples
python debug_tiling_example.py
```

This will:
- Analyze constraint models
- Create text and HTML visualizations
- Demonstrate error handling
- Compare different tiling strategies

## New Debugging Features

### 🔍 TileConstraintDebugger

**Features**:
- Variable analysis (ranges, fixed values, tileability)
- Constraint cataloging (geometrical, memory, policy)
- Automatic issue detection (over-constraints, tight memory)
- User-friendly error explanations with suggestions
- Text-based tiling visualization
- Solution comparison

**When to use**:
- Developing new TileConstraints
- Debugging "No solution found" errors
- Understanding why certain variables can't be tiled
- Comparing different tiling strategies

### 🎨 TileConstraintVisualizer

**Features**:
- Interactive HTML visualizations
- 2D spatial tiling pattern visualization with color-coded tiles
- Memory usage per tile (bar charts)
- Detailed tile information tables
- Beautiful, professional-looking reports
- Hover tooltips for tile details (in HTML)

**When to use**:
- Understanding complex tiling patterns
- Presenting tiling strategies to team
- Documentation and reports
- Debugging spatial tiling issues
- Analyzing memory usage patterns

### 📊 What Gets Visualized

#### 1. Variable Analysis
```
📊 VARIABLES:
────────────────────────────────────────────────────────────────────────────────
  ✓ input_height_dim_1                       : 1..64                (range: 63)
  ✓ input_width_dim_2                        : 1..64                (range: 63)
  ⚠ input_channels_dim_3                     : 32..32               (range: 0)
```

#### 2. Constraint Overview
```
📝 CONSTRAINTS (12):
────────────────────────────────────────────────────────────────────────────────
   1. [geometrical        ] (output_height_dim_1 == ((input_height_dim_1 - 3) / 1))
   2. [geometrical        ] (output_width_dim_2 == ((input_width_dim_2 - 3) / 1))
   3. [policy             ] (input_channels_dim_3 == 32)
```

#### 3. Memory Constraints
```
💾 MEMORY CONSTRAINTS:
────────────────────────────────────────────────────────────────────────────────
  L1              (size:      16384 bytes)
    (input_size + weight_size + output_size <= 16384)
```

#### 4. 2D Tiling Pattern (HTML)

The HTML visualization shows:
- **Color-coded tiles**: Each tile has a unique color
- **SVG graphics**: Scalable, interactive visualization
- **Hover information**: Tile number, offset, size
- **Grid overlay**: Shows tensor boundaries
- **Statistics**: Number of tiles, memory usage, etc.

#### 5. Tile Details Table

| Tile # | Input (offset → dims) | Output (offset → dims) | Variables |
|--------|----------------------|------------------------|-----------|
| 0 | `(0,0,0,0) → (1,17,17,16)` | `(0,0,0,0) → (1,15,15,32)` | `padding_top=1`<br>`padding_left=1` |
| 1 | `(0,0,15,0) → (1,17,17,16)` | `(0,0,15,0) → (1,15,15,32)` | `padding_top=1`<br>`padding_left=0` |

## Common Use Cases

### Use Case 1: "No Solution Found" Error

```python
try:
    collector = tilerModel.trySolveModel()
except Exception as e:
    # Use debugger to explain
    debugger = TileConstraintDebugger()
    explanation = debugger.explain_constraint_failure(str(e), tilerModel)
    print(explanation)

    # Analyze the model
    debugger.analyze_constraint_model(tilerModel, "MyOperator")
```

**Output provides**:
- Possible causes (over-constrained, memory too small)
- Debugging steps
- Suggestions for fixes
- Model statistics

### Use Case 2: Understanding a Complex Tiling Pattern

```python
# Generate HTML visualization
visualizer = TileConstraintVisualizer()
html_path = visualizer.visualize_tiling_schedule(
    tilingSchedule=schedule,
    variableReplacement=variables,
    operatorName="Conv2D",
    tensorShapes=shapes
)

# Open in browser to see:
# - 2D tiling pattern with colors
# - Memory usage per tile
# - Detailed tile information
# - Interactive hover tooltips
```

### Use Case 3: Comparing Tiling Strategies

```python
# Generate two solutions
solution_small_mem = generateTiling(l1_size=8192)
solution_large_mem = generateTiling(l1_size=16384)

# Compare
debugger = TileConstraintDebugger()
debugger.compare_tiling_solutions(
    solution1=solution_small_mem,
    solution2=solution_large_mem,
    names=("L1=8KB", "L1=16KB")
)

# Visualize both
visualizer = TileConstraintVisualizer()
visualizer.visualize_tiling_schedule(..., output_file="small_mem.html")
visualizer.visualize_tiling_schedule(..., output_file="large_mem.html")
```

### Use Case 4: Developing New TileConstraints

```python
class MyNewTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel, parseDict, ctxt):
        # Add your constraints
        tilerModel.addConstraint(...)

        # Debug inline during development
        from Deeploy.TilingExtension.TileConstraintDebugger import debug_tile_constraint
        debug_tile_constraint(
            tileConstraintClass=MyNewTileConstraint,
            parseDict=parseDict,
            ctxt=ctxt,
            tilerModel=tilerModel,
            verbose=True  # Detailed output during development
        )

        return tilerModel
```

## Architecture

### Constraint Solving Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Define TileConstraints                                      │
│    - Geometrical: output_h = (input_h - kernel) / stride       │
│    - Policy: input_channels must be whole                      │
│    - Memory: tiles must fit in L1                              │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Build TilerModel (OR-Tools CSP)                             │
│    - Variables: tile dimensions                                │
│    - Constraints: equations and inequalities                   │
│    - Objective: maximize tile size                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Solve (OR-Tools CP-SAT Solver)                              │
│    - Find valid tile dimensions                                │
│    - Satisfy all constraints                                   │
│    - Optimize objective                                        │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Serialize to TilingSchedule                                 │
│    - Compute input/output cubes                                │
│    - Generate memory addresses                                 │
│    - Create variable replacements                              │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Generate Code                                               │
│    - DMA transfers                                             │
│    - Loop over tiles                                           │
│    - Kernel invocations                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Where Debugging Fits In

```
TilerModel.trySolveModel()
    │
    ├─ Before solving: TileConstraintDebugger.analyze_constraint_model()
    │  → Shows variables, constraints, potential issues
    │
    ├─ If solving fails: TileConstraintDebugger.explain_constraint_failure()
    │  → Explains error, suggests fixes
    │
    └─ After solving: TileConstraintVisualizer.visualize_tiling_schedule()
       → Shows tiling pattern, memory usage, tile details
```

## Best Practices

1. **Debug Early**: Add debugging during constraint development, not just when errors occur

2. **Use Verbose Mode**: Enable `verbose=True` during development for detailed output

3. **Visualize Often**: Create HTML visualizations to understand tiling patterns

4. **Save Visualizations**: Keep HTML files for documentation and future reference

5. **Compare Strategies**: Use comparison tools to understand trade-offs

6. **Document Constraints**: Add comments explaining why each constraint exists

7. **Test Edge Cases**: Use debugger to verify edge tiles, remainder tiles, single-tile cases

## Integration with Existing Code

The debugging tools are **non-intrusive**:
- They don't modify the tiling flow
- They can be added/removed without affecting functionality
- They work with any TileConstraint implementation
- They're optional (for debugging only)

### Minimal Integration

Just import and use when needed:

```python
# Your existing code
tilerModel = self.addGeometricalConstraint(tilerModel, parseDict, ctxt)
tilerModel = self.addPolicyConstraint(tilerModel, parseDict, ctxt)

# Add debugging (optional)
if DEBUG:
    from Deeploy.TilingExtension.TileConstraintDebugger import TileConstraintDebugger
    debugger = TileConstraintDebugger()
    debugger.analyze_constraint_model(tilerModel, parseDict['nodeName'])

# Continue with existing code
collector = tilerModel.trySolveModel()
```

## File Structure

```
TilingExtension/
├── README.md                          ← You are here
├── DEBUGGING_GUIDE.md                 ← Detailed debugging guide
│
├── TileConstraint.py                  ← Base class
├── TilerModel.py                      ← Constraint solver
├── TilingCodegen.py                   ← Code generation
├── MemoryConstraints.py               ← Memory management
├── MemoryScheduler.py                 ← DMA scheduling
├── AsyncDma.py                        ← Async DMA handling
│
├── TileConstraintDebugger.py          ← NEW: Debugging tools
├── TileConstraintVisualizer.py        ← NEW: HTML visualizations
│
└── examples/
    └── debug_tiling_example.py        ← NEW: Example usage
```

## Resources

### Documentation
- **[CLAUDE.md](../../CLAUDE.md)**: Complete tiling constraints tutorial
- **[DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md)**: Debugging workflow and examples
- **[Deeploy Docs](https://pulp-platform.github.io/Deeploy/)**: Official documentation

### Code Examples
- **[debug_tiling_example.py](./examples/debug_tiling_example.py)**: Runnable examples
- **Target TileConstraints**: See `Deeploy/Targets/*/TileConstraints/` for real-world examples

### External Resources
- **[OR-Tools](https://developers.google.com/optimization/cp)**: Constraint programming
- **[PULP Platform](https://pulp-platform.org/)**: Target hardware documentation

## Contributing

### Adding New Debug Features

1. Add functionality to `TileConstraintDebugger.py` or `TileConstraintVisualizer.py`
2. Update `DEBUGGING_GUIDE.md` with examples
3. Add tests if applicable
4. Create example in `examples/`

### Reporting Issues

If debugging tools don't work as expected:
1. Check if it's a constraint issue (use the tools to debug)
2. Report bugs on GitHub with example code
3. Suggest improvements via pull requests

## FAQ

**Q: Do I need to use these tools?**
A: No, they're optional debugging aids. The core tiling flow works without them.

**Q: Do the debugging tools slow down compilation?**
A: Only if you use them. They're opt-in and add minimal overhead when used.

**Q: Can I use these tools with existing TileConstraints?**
A: Yes! They work with any TileConstraint without modification.

**Q: What browsers work with HTML visualizations?**
A: Modern browsers (Chrome, Firefox, Safari, Edge). No special plugins needed.

**Q: Can I customize the visualizations?**
A: Yes! The HTML generator code is in `TileConstraintVisualizer.py`. Modify as needed.

**Q: How do I integrate this with CI/CD?**
A: See the "Integration with CI/CD" section in `DEBUGGING_GUIDE.md`.

---

**Need Help?**
- Read [CLAUDE.md](../../CLAUDE.md) for tiling fundamentals
- Read [DEBUGGING_GUIDE.md](./DEBUGGING_GUIDE.md) for debugging workflows
- Run [examples](./examples/debug_tiling_example.py) to see tools in action
- Ask questions on GitHub Issues

**Happy Tiling! 🔲**
