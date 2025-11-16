# Tiling Constraint Debugging Guide

This guide explains how to use Deeploy's tiling constraint debugging and visualization tools.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Tools Overview](#tools-overview)
3. [Debugging Workflow](#debugging-workflow)
4. [Examples](#examples)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Import the Debugger

```python
from Deeploy.TilingExtension.TileConstraintDebugger import TileConstraintDebugger, debug_tile_constraint
```

### Step 2: Analyze Your Constraints

Add debugging to your TileConstraint:

```python
@staticmethod
def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict,
                             ctxt: NetworkContext) -> TilerModel:
    # Add your constraints
    tilerModel.addConstraint(outputHeight == inputHeight // 2)

    # DEBUG: Analyze the constraints
    debugger = TileConstraintDebugger(verbose=True)
    debugger.analyze_constraint_model(tilerModel, parseDict.get('nodeName', 'MyOp'))

    return tilerModel
```

### Step 3: Visualize Tiling Solutions

```python
from Deeploy.TilingExtension.TileConstraintVisualizer import TileConstraintVisualizer

# After obtaining a tiling solution
visualizer = TileConstraintVisualizer(output_dir="./my_visualizations")

html_path = visualizer.visualize_tiling_schedule(
    tilingSchedule=tilingSchedule,
    variableReplacement=variableReplacementScheme,
    operatorName="Conv2D_layer1",
    tensorShapes={
        'data_in': (1, 32, 32, 64),
        'data_out': (1, 30, 30, 128)
    }
)

print(f"Open {html_path} in your browser to see the visualization")
```

---

## Tools Overview

### 1. TileConstraintDebugger

**Purpose**: Analyze constraint models and identify issues

**Key Features**:
- Variable analysis (ranges, fixed values)
- Constraint cataloging (geometrical, memory, policy, performance hints)
- Warning detection (over-constrained variables, tight memory)
- Helpful error explanations

**Main Methods**:

```python
debugger = TileConstraintDebugger(verbose=True)

# Analyze a constraint model
analysis = debugger.analyze_constraint_model(
    tilerModel=tilerModel,
    operatorName="MyOperator"
)

# Visualize a tiling solution (text-based)
debugger.visualize_tiling_solution(
    tilingSchedule=tilingSchedule,
    variableReplacement=variableReplacementScheme,
    operatorName="MyOperator",
    tensorShapes=shapes
)

# Explain constraint failures
explanation = debugger.explain_constraint_failure(
    error_msg=str(exception),
    tilerModel=tilerModel
)
print(explanation)
```

### 2. TileConstraintVisualizer

**Purpose**: Create interactive HTML visualizations

**Key Features**:
- 2D spatial tiling pattern visualization
- Per-tile memory usage charts
- Interactive tile information tables
- Beautiful, professional-looking reports

**Main Methods**:

```python
visualizer = TileConstraintVisualizer(output_dir="./viz")

# Create HTML visualization
html_path = visualizer.visualize_tiling_schedule(
    tilingSchedule=schedule,
    variableReplacement=var_replacement,
    operatorName="Conv2D",
    tensorShapes=shapes
)
```

### 3. Convenience Function: debug_tile_constraint

**Purpose**: Quick debugging during constraint development

```python
from Deeploy.TilingExtension.TileConstraintDebugger import debug_tile_constraint

@staticmethod
def addGeometricalConstraint(tilerModel, parseDict, ctxt):
    # Your constraints
    tilerModel.addConstraint(...)

    # Quick debug
    debug_tile_constraint(
        tileConstraintClass=MyConstraintClass,
        parseDict=parseDict,
        ctxt=ctxt,
        tilerModel=tilerModel,
        verbose=True
    )

    return tilerModel
```

---

## Debugging Workflow

### Scenario 1: "No solution found" Error

**Steps**:

1. **Analyze the constraint model**:
```python
debugger = TileConstraintDebugger(verbose=True)
analysis = debugger.analyze_constraint_model(tilerModel, "MyOp")
```

2. **Look for issues in the output**:
   - Variables with zero range (cannot be tiled)
   - Very small variable ranges (limited tiling options)
   - Many constraints (possible over-constraint)

3. **Check memory constraints**:
   - Look at the "MEMORY CONSTRAINTS" section
   - Try increasing memory size

4. **Relax performance hints**:
```python
# Instead of:
tilerModel.addConstraint(var == maxValue)

# Use:
tilerModel.addConstraint(var == maxValue, strategy=PerformanceHint(1))
```

5. **Test constraints individually**:
   - Comment out constraints one by one
   - Find which constraint causes infeasibility

### Scenario 2: Understanding a Tiling Solution

**Steps**:

1. **Generate HTML visualization**:
```python
visualizer = TileConstraintVisualizer()
html_path = visualizer.visualize_tiling_schedule(...)
```

2. **Open the HTML file in a browser**

3. **Analyze the visualization**:
   - Check the 2D tiling pattern
   - Verify tile sizes are reasonable
   - Look for irregular patterns (may indicate remainder tiles)
   - Check memory usage per tile

4. **Use text-based visualization for quick checks**:
```python
debugger = TileConstraintDebugger()
debugger.visualize_tiling_solution(...)  # Prints to console
```

### Scenario 3: Comparing Different Tiling Strategies

**Steps**:

1. **Generate two different solutions** (e.g., different memory sizes):
```python
# Solution 1: Small memory
solution1 = generateTilingSolution(l1_size=8192)

# Solution 2: Large memory
solution2 = generateTilingSolution(l1_size=16384)
```

2. **Compare them**:
```python
debugger = TileConstraintDebugger()
debugger.compare_tiling_solutions(
    solution1=solution1,
    solution2=solution2,
    names=("L1=8KB", "L1=16KB")
)
```

3. **Visualize both**:
```python
visualizer = TileConstraintVisualizer()
visualizer.visualize_tiling_schedule(..., output_file="small_mem.html")
visualizer.visualize_tiling_schedule(..., output_file="large_mem.html")
```

### Scenario 4: Debugging Incorrect Results

**Steps**:

1. **Visualize the tiling solution**:
```python
debugger = TileConstraintDebugger()
debugger.visualize_tiling_solution(...)
```

2. **Check tile boundaries**:
   - Verify offset calculations
   - Check for gaps or overlaps between tiles
   - Ensure edge tiles are handled correctly

3. **Verify variable replacements**:
   - Look at the per-tile variable values
   - Check if padding is correct for edge tiles
   - Verify stride calculations

4. **Add logging to serializeTilingSolution**:
```python
@classmethod
def serializeTilingSolution(cls, ...):
    for i, (inCube, outCube) in enumerate(zip(inputCubes, outputCubes)):
        print(f"Tile {i}:")
        print(f"  Input: offset={inCube.offset}, dims={inCube.dims}")
        print(f"  Output: offset={outCube.offset}, dims={outCube.dims}")
        # Verify input-output relationship manually
```

---

## Examples

### Example 1: Debugging a Custom Conv2D Constraint

```python
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TileConstraintDebugger import TileConstraintDebugger

class MyConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel, parseDict, ctxt):
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputH = tilerModel.getTensorDimVar(inputBufferName, 1)
        inputW = tilerModel.getTensorDimVar(inputBufferName, 2)
        outputH = tilerModel.getTensorDimVar(outputBufferName, 1)
        outputW = tilerModel.getTensorDimVar(outputBufferName, 2)

        kernel_size = parseDict['kernel_size']
        stride = parseDict['stride']

        # Add constraints
        tilerModel.addConstraint(outputH == (inputH - kernel_size) // stride + 1)
        tilerModel.addConstraint(outputW == (inputW - kernel_size) // stride + 1)

        # DEBUG: Analyze constraints
        debugger = TileConstraintDebugger(verbose=True)
        analysis = debugger.analyze_constraint_model(
            tilerModel,
            parseDict.get('nodeName', 'MyConv2D')
        )

        # Check for warnings
        if analysis['warnings']:
            print("⚠️ WARNINGS DETECTED:")
            for warning in analysis['warnings']:
                print(f"  {warning}")

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel, parseDict, ctxt):
        inputBuffer = ctxt.lookup(parseDict['data_in'])

        inputH = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        inputW = tilerModel.getTensorDimVar(inputBuffer.name, 2)

        # Input tiles must be at least kernel size
        tilerModel.addConstraint(inputH >= parseDict['kernel_size'])
        tilerModel.addConstraint(inputW >= parseDict['kernel_size'])

        # Input tiles must be divisible by stride
        tilerModel.addConstraint((inputH % parseDict['stride']) == 0)
        tilerModel.addConstraint((inputW % parseDict['stride']) == 0)

        # DEBUG: Re-analyze with policy constraints
        debugger = TileConstraintDebugger(verbose=True)
        debugger.analyze_constraint_model(
            tilerModel,
            parseDict.get('nodeName', 'MyConv2D_with_policy')
        )

        return tilerModel
```

### Example 2: Visualizing a Complete Tiling Solution

```python
from Deeploy.TilingExtension.TileConstraintVisualizer import TileConstraintVisualizer

# Assume we have a tiling solution
# (This would come from your actual tiling flow)

tensorShapes = {
    'data_in': (1, 64, 64, 32),    # NHWC
    'weight': (64, 3, 3, 32),       # OC, H, W, IC
    'data_out': (1, 62, 62, 64)     # NHWC
}

# Create visualizer
visualizer = TileConstraintVisualizer(output_dir="./conv2d_visualization")

# Generate visualization
html_path = visualizer.visualize_tiling_schedule(
    tilingSchedule=myTilingSchedule,
    variableReplacement=myVariableReplacement,
    operatorName="Conv2D_layer1",
    tensorShapes=tensorShapes
)

print(f"✓ Visualization saved to: {html_path}")
print("  Open this file in your web browser to explore the tiling pattern")
```

### Example 3: Handling and Explaining Errors

```python
from Deeploy.TilingExtension.TileConstraintDebugger import TileConstraintDebugger

try:
    # Try to solve the tiling problem
    collector = tilerModel.trySolveModel()
except Exception as e:
    # Use debugger to explain the error
    debugger = TileConstraintDebugger()
    explanation = debugger.explain_constraint_failure(
        error_msg=str(e),
        tilerModel=tilerModel
    )

    print(explanation)

    # Also show constraint analysis
    debugger.analyze_constraint_model(tilerModel, "FailedOperator")

    # Suggestions will be in the output
```

### Example 4: Comparing Tiling Strategies

```python
from Deeploy.TilingExtension.TileConstraintDebugger import TileConstraintDebugger
from Deeploy.TilingExtension.TileConstraintVisualizer import TileConstraintVisualizer

# Generate two solutions with different memory sizes
solution_8kb = generateTiling(l1_size=8192)
solution_16kb = generateTiling(l1_size=16384)

# Compare numerically
debugger = TileConstraintDebugger()
debugger.compare_tiling_solutions(
    solution1=solution_8kb,
    solution2=solution_16kb,
    names=("L1=8KB", "L1=16KB")
)

# Visualize both
visualizer = TileConstraintVisualizer()

visualizer.visualize_tiling_schedule(
    *solution_8kb,
    operatorName="Conv2D",
    tensorShapes=shapes,
    output_file="conv2d_8kb.html"
)

visualizer.visualize_tiling_schedule(
    *solution_16kb,
    operatorName="Conv2D",
    tensorShapes=shapes,
    output_file="conv2d_16kb.html"
)

print("✓ Open both HTML files side-by-side to compare")
```

---

## Troubleshooting

### Issue: Debugger shows no warnings but solver still fails

**Solution**: The debugger analyzes the model *before* solving. Failures during solving may be due to:
- Memory constraints that are too tight
- Complex constraint interactions not visible from variable ranges

**Try**:
1. Increase memory size incrementally
2. Remove performance hints
3. Check for division-by-zero or modulo with variables that can be zero

### Issue: 2D visualization doesn't appear

**Possible causes**:
- Tensors don't have spatial dimensions (need at least 3D)
- Layout is not NHWC (debugger assumes NHWC for spatial dims)

**Solution**: Ensure your tensors follow NHWC layout and have H, W dimensions

### Issue: HTML visualization is too large

**Solution**: The visualization shows all tiles. For very large numbers of tiles:
- Limit the visualization to first N tiles
- Use text-based visualization instead
- Increase browser memory limit

### Issue: Variable replacements show "None"

**Cause**: `perTileReplacements` dictionary is empty

**Solution**: Ensure `serializeTilingSolution` populates the replacements dictionary:
```python
replacements = {
    "my_var": [value for each tile],
    ...
}
```

---

## Best Practices

1. **Debug Early**: Add debugging to constraints during development, not just when things fail

2. **Use Verbose Mode**: Set `verbose=True` to see detailed output during development

3. **Save Visualizations**: Keep HTML visualizations for documentation and debugging later

4. **Compare Solutions**: Always compare different tiling strategies to understand trade-offs

5. **Document Constraints**: Add comments explaining *why* each constraint exists

6. **Test Edge Cases**: Use debugger to verify edge tiles, remainder tiles, and single-tile cases

---

## Integration with CI/CD

You can integrate visualization into your test suite:

```python
def test_conv2d_tiling():
    # Generate tiling solution
    solution = generateConv2DTiling(...)

    # Create visualization for debugging
    if os.environ.get('SAVE_VISUALIZATIONS'):
        visualizer = TileConstraintVisualizer(output_dir="./test_visualizations")
        visualizer.visualize_tiling_schedule(
            *solution,
            operatorName=f"test_conv2d_{test_id}",
            tensorShapes=shapes
        )

    # Run assertions
    assert verify_tiling_correctness(solution)
```

Then run tests with: `SAVE_VISUALIZATIONS=1 pytest`

---

## Additional Resources

- **CLAUDE.md**: Comprehensive guide to tiling constraints
- **Deeploy Documentation**: https://pulp-platform.github.io/Deeploy/
- **OR-Tools Documentation**: https://developers.google.com/optimization/cp

---

## Feedback and Contributions

If you find bugs or have suggestions for improving these tools, please:
1. Open an issue on GitHub
2. Submit a pull request with improvements
3. Share your debugging experiences with the community

---

*Happy Debugging! 🐛🔍*
