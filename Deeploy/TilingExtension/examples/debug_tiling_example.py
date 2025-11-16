#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
Example script demonstrating the use of TileConstraint debugging tools.

This script shows how to:
1. Debug constraint models
2. Visualize tiling solutions
3. Handle errors gracefully
4. Compare different tiling strategies
"""

import sys
import os
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from Deeploy.TilingExtension.TileConstraintDebugger import TileConstraintDebugger, debug_tile_constraint
from Deeploy.TilingExtension.TileConstraintVisualizer import TileConstraintVisualizer
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t


def example_1_analyze_constraints():
    """
    Example 1: Analyze a constraint model to understand variable ranges and constraints.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Analyzing Constraint Models")
    print("="*80 + "\n")

    # Create a simple tiling model
    tilerModel = TilerModel()

    # Add some example variables (simulating a Conv2D operation)
    input_h = tilerModel.addVariable("input_height", lowerBound=1, upperBound=64)
    input_w = tilerModel.addVariable("input_width", lowerBound=1, upperBound=64)
    input_c = tilerModel.addVariable("input_channels", lowerBound=32, upperBound=32)  # Fixed

    output_h = tilerModel.addVariable("output_height", lowerBound=1, upperBound=62)
    output_w = tilerModel.addVariable("output_width", lowerBound=1, upperBound=62)
    output_c = tilerModel.addVariable("output_channels", lowerBound=64, upperBound=64)  # Fixed

    # Add geometrical constraints
    kernel_size = 3
    stride = 1
    tilerModel.addConstraint(output_h == (input_h - kernel_size) // stride + 1)
    tilerModel.addConstraint(output_w == (input_w - kernel_size) // stride + 1)

    # Add policy constraints
    tilerModel.addConstraint(input_c == 32)  # Keep channels whole
    tilerModel.addConstraint(output_c == 64)
    tilerModel.addConstraint(input_h >= kernel_size)  # Minimum tile size
    tilerModel.addConstraint(input_w >= kernel_size)

    # Analyze the model
    debugger = TileConstraintDebugger(verbose=True)
    analysis = debugger.analyze_constraint_model(tilerModel, "Example_Conv2D")

    print("\n✓ Analysis complete!")
    print(f"  - Found {len(analysis['variables'])} variables")
    print(f"  - Found {len(analysis['constraints'])} constraints")
    print(f"  - Warnings: {len(analysis['warnings'])}")
    print(f"  - Suggestions: {len(analysis['suggestions'])}")


def example_2_visualize_tiling_solution():
    """
    Example 2: Create and visualize a tiling solution.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Visualizing Tiling Solutions")
    print("="*80 + "\n")

    # Create a mock tiling solution for a Conv2D operation
    # Tensor shapes: Input (1, 32, 32, 16), Output (1, 30, 30, 32)

    # Define 4 tiles in a 2x2 pattern
    inputLoadSchedule = [
        {"data_in": HyperRectangle(offset=(0, 0, 0, 0), dims=(1, 17, 17, 16))},    # Tile 0: top-left
        {"data_in": HyperRectangle(offset=(0, 0, 15, 0), dims=(1, 17, 17, 16))},   # Tile 1: top-right
        {"data_in": HyperRectangle(offset=(0, 15, 0, 0), dims=(1, 17, 17, 16))},   # Tile 2: bottom-left
        {"data_in": HyperRectangle(offset=(0, 15, 15, 0), dims=(1, 17, 17, 16))},  # Tile 3: bottom-right
    ]

    outputLoadSchedule = [
        {"data_out": HyperRectangle(offset=(0, 0, 0, 0), dims=(1, 15, 15, 32))},   # Tile 0
        {"data_out": HyperRectangle(offset=(0, 0, 15, 0), dims=(1, 15, 15, 32))},  # Tile 1
        {"data_out": HyperRectangle(offset=(0, 15, 0, 0), dims=(1, 15, 15, 32))},  # Tile 2
        {"data_out": HyperRectangle(offset=(0, 15, 15, 0), dims=(1, 15, 15, 32))}, # Tile 3
    ]

    inputBaseOffsets = {"data_in": [0x0000]}
    outputBaseOffsets = {"data_out": [0x4000]}

    tilingSchedule = TilingSchedule(
        inputBaseOffsets=inputBaseOffsets,
        outputBaseOffsets=outputBaseOffsets,
        inputLoadSchedule=inputLoadSchedule,
        outputLoadSchedule=outputLoadSchedule
    )

    # Variable replacements (tile-specific parameters)
    replacements = {
        "tile_height": [17, 17, 17, 17],
        "tile_width": [17, 17, 17, 17],
        "output_height": [15, 15, 15, 15],
        "output_width": [15, 15, 15, 15],
        "padding_top": [1, 1, 0, 0],
        "padding_left": [1, 0, 1, 0],
    }

    replacementTypes = {k: PointerClass(uint16_t) for k in replacements.keys()}

    variableReplacement = VariableReplacementScheme(
        perTileReplacements=replacements,
        replacementTypes=replacementTypes
    )

    tensorShapes = {
        'data_in': (1, 32, 32, 16),
        'data_out': (1, 30, 30, 32)
    }

    # Text-based visualization
    print("Text-based visualization:")
    debugger = TileConstraintDebugger(verbose=False)
    debugger.visualize_tiling_solution(
        tilingSchedule=tilingSchedule,
        variableReplacement=variableReplacement,
        operatorName="Example_Conv2D",
        tensorShapes=tensorShapes
    )

    # HTML visualization
    print("\nCreating HTML visualization...")
    visualizer = TileConstraintVisualizer(output_dir="./example_visualizations")
    html_path = visualizer.visualize_tiling_schedule(
        tilingSchedule=tilingSchedule,
        variableReplacement=variableReplacement,
        operatorName="Example_Conv2D",
        tensorShapes=tensorShapes,
        output_file="example_conv2d.html"
    )

    print(f"\n✓ HTML visualization created: {html_path}")
    print("  Open this file in your web browser to explore the interactive visualization")


def example_3_error_handling():
    """
    Example 3: Handle and explain constraint errors.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Error Handling and Explanations")
    print("="*80 + "\n")

    # Create a model with contradictory constraints
    tilerModel = TilerModel()

    var_x = tilerModel.addVariable("x", lowerBound=1, upperBound=100)
    var_y = tilerModel.addVariable("y", lowerBound=1, upperBound=100)

    # Contradictory constraints
    tilerModel.addConstraint(var_x >= 50)
    tilerModel.addConstraint(var_x <= 10)  # Contradiction!

    debugger = TileConstraintDebugger(verbose=True)

    # Analyze the model first
    print("Analyzing the (intentionally broken) model:")
    analysis = debugger.analyze_constraint_model(tilerModel, "Broken_Example")

    # Simulate an error
    error_msg = "Error in Tiler: No solution found"

    print("\nExplaining the error:")
    explanation = debugger.explain_constraint_failure(error_msg, tilerModel)
    print(explanation)


def example_4_compare_solutions():
    """
    Example 4: Compare two different tiling solutions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Comparing Tiling Solutions")
    print("="*80 + "\n")

    # Solution 1: Aggressive tiling (4 tiles)
    schedule1 = TilingSchedule(
        inputBaseOffsets={"data_in": [0x0000]},
        outputBaseOffsets={"data_out": [0x2000]},
        inputLoadSchedule=[
            {"data_in": HyperRectangle((0, 0, 0, 0), (1, 17, 17, 16))},
            {"data_in": HyperRectangle((0, 0, 15, 0), (1, 17, 17, 16))},
            {"data_in": HyperRectangle((0, 15, 0, 0), (1, 17, 17, 16))},
            {"data_in": HyperRectangle((0, 15, 15, 0), (1, 17, 17, 16))},
        ],
        outputLoadSchedule=[
            {"data_out": HyperRectangle((0, 0, 0, 0), (1, 15, 15, 32))},
            {"data_out": HyperRectangle((0, 0, 15, 0), (1, 15, 15, 32))},
            {"data_out": HyperRectangle((0, 15, 0, 0), (1, 15, 15, 32))},
            {"data_out": HyperRectangle((0, 15, 15, 0), (1, 15, 15, 32))},
        ]
    )
    var_repl1 = VariableReplacementScheme({}, {})

    # Solution 2: Less aggressive tiling (2 tiles)
    schedule2 = TilingSchedule(
        inputBaseOffsets={"data_in": [0x0000]},
        outputBaseOffsets={"data_out": [0x4000]},
        inputLoadSchedule=[
            {"data_in": HyperRectangle((0, 0, 0, 0), (1, 32, 17, 16))},
            {"data_in": HyperRectangle((0, 0, 15, 0), (1, 32, 17, 16))},
        ],
        outputLoadSchedule=[
            {"data_out": HyperRectangle((0, 0, 0, 0), (1, 30, 15, 32))},
            {"data_out": HyperRectangle((0, 0, 15, 0), (1, 30, 15, 32))},
        ]
    )
    var_repl2 = VariableReplacementScheme({}, {})

    # Compare
    debugger = TileConstraintDebugger(verbose=False)
    debugger.compare_tiling_solutions(
        solution1=(var_repl1, schedule1),
        solution2=(var_repl2, schedule2),
        names=("Aggressive (4 tiles)", "Conservative (2 tiles)")
    )

    print("\n✓ Comparison complete!")
    print("  Solution 1 uses more tiles but smaller memory per tile")
    print("  Solution 2 uses fewer tiles but larger memory per tile")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" "*20 + "TILING DEBUGGING TOOLS - EXAMPLES")
    print("="*80)

    try:
        example_1_analyze_constraints()
    except Exception as e:
        print(f"\n❌ Example 1 failed: {e}")

    try:
        example_2_visualize_tiling_solution()
    except Exception as e:
        print(f"\n❌ Example 2 failed: {e}")

    try:
        example_3_error_handling()
    except Exception as e:
        print(f"\n❌ Example 3 failed: {e}")

    try:
        example_4_compare_solutions()
    except Exception as e:
        print(f"\n❌ Example 4 failed: {e}")

    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
