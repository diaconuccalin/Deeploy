# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
TileConstraintDebugger: Tools for debugging and understanding tiling constraints

This module provides user-friendly debugging tools for Deeploy's tiling system:
- Detailed constraint analysis
- Visual representations of tiling solutions
- Interactive exploration of constraint interactions
- Enhanced error messages with suggestions
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from ortools.constraint_solver.pywrapcp import IntExpr

from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation

log = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints in the tiling system"""
    GEOMETRICAL = "geometrical"
    MEMORY = "memory"
    POLICY = "policy"
    PERFORMANCE_HINT = "performance_hint"


@dataclass
class ConstraintInfo:
    """Information about a single constraint"""
    constraint_type: ConstraintType
    expression: str
    description: str
    is_satisfied: Optional[bool] = None
    related_variables: List[str] = None

    def __post_init__(self):
        if self.related_variables is None:
            self.related_variables = []


class TileConstraintDebugger:
    """
    Main debugger class for analyzing tiling constraints.

    Features:
    - Analyze constraint conflicts
    - Visualize tiling solutions
    - Provide helpful error messages
    - Suggest fixes for common issues
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the debugger.

        Args:
            verbose: If True, print detailed information during analysis
        """
        self.verbose = verbose
        self.constraints: List[ConstraintInfo] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []

    def analyze_constraint_model(self, tilerModel: TilerModel, operatorName: str = "Unknown") -> Dict[str, Any]:
        """
        Analyze a TilerModel and provide detailed diagnostics.

        Args:
            tilerModel: The constraint model to analyze
            operatorName: Name of the operator for context

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'operator': operatorName,
            'variables': self._analyze_variables(tilerModel),
            'constraints': self._analyze_constraints(tilerModel),
            'memory_constraints': self._analyze_memory_constraints(tilerModel),
            'warnings': [],
            'suggestions': []
        }

        # Check for common issues
        self._check_common_issues(tilerModel, analysis)

        if self.verbose:
            self._print_analysis(analysis)

        return analysis

    def _analyze_variables(self, tilerModel: TilerModel) -> Dict[str, Dict[str, int]]:
        """Extract and analyze all variables in the model"""
        variables = {}

        for varName, var in tilerModel._variables.items():
            variables[varName] = {
                'min': var.Min(),
                'max': var.Max(),
                'range': var.Max() - var.Min()
            }

        return variables

    def _analyze_constraints(self, tilerModel: TilerModel) -> List[Dict[str, Any]]:
        """Analyze all constraints in the model"""
        constraints = []

        for constraint in tilerModel._constraints:
            constraints.append({
                'expression': constraint.DebugString(),
                'type': 'geometrical'
            })

        for priority, constraint in tilerModel._performanceConstraints:
            constraints.append({
                'expression': constraint.DebugString(),
                'type': 'performance_hint',
                'priority': priority
            })

        return constraints

    def _analyze_memory_constraints(self, tilerModel: TilerModel) -> List[Dict[str, Any]]:
        """Analyze memory-related constraints"""
        memory_constraints = []

        for memLevel, constraint in tilerModel._memoryConstraints:
            memory_constraints.append({
                'memory_level': memLevel.name,
                'memory_size': memLevel.size,
                'constraint': constraint.DebugString(),
                'type': 'memory'
            })

        for priority, (memLevel, constraint) in tilerModel._performanceMemoryConstraints:
            memory_constraints.append({
                'memory_level': memLevel.name,
                'memory_size': memLevel.size,
                'constraint': constraint.DebugString(),
                'type': 'memory_performance_hint',
                'priority': priority
            })

        return memory_constraints

    def _check_common_issues(self, tilerModel: TilerModel, analysis: Dict[str, Any]):
        """Check for common constraint issues and add warnings/suggestions"""

        # Check 1: Variables with zero range
        for varName, varInfo in analysis['variables'].items():
            if varInfo['range'] == 0:
                analysis['warnings'].append(
                    f"⚠️  Variable '{varName}' has zero range (fixed at {varInfo['max']}). "
                    f"This dimension cannot be tiled."
                )

        # Check 2: Very small variable ranges
        for varName, varInfo in analysis['variables'].items():
            if 0 < varInfo['range'] < 4 and '_dim_' in varName:
                analysis['warnings'].append(
                    f"⚠️  Variable '{varName}' has very small range ({varInfo['min']}-{varInfo['max']}). "
                    f"Limited tiling options available."
                )

        # Check 3: Many performance hints
        perf_hints = [c for c in analysis['constraints'] if c['type'] == 'performance_hint']
        if len(perf_hints) > 5:
            analysis['suggestions'].append(
                f"💡 Found {len(perf_hints)} performance hints. Consider reducing them if solver is slow."
            )

        # Check 4: Memory constraints
        for mc in analysis['memory_constraints']:
            if mc['type'] == 'memory':
                analysis['suggestions'].append(
                    f"💡 Memory constraint on {mc['memory_level']} (size: {mc['memory_size']} bytes). "
                    f"Increase memory size if 'no solution found' errors occur."
                )

    def _print_analysis(self, analysis: Dict[str, Any]):
        """Pretty-print the analysis results"""
        print("\n" + "=" * 80)
        print(f"  TILING CONSTRAINT ANALYSIS: {analysis['operator']}")
        print("=" * 80)

        # Variables
        print("\n📊 VARIABLES:")
        print("-" * 80)
        for varName, varInfo in sorted(analysis['variables'].items()):
            range_str = f"{varInfo['min']}..{varInfo['max']}"
            status = "✓" if varInfo['range'] > 0 else "⚠"
            print(f"  {status} {varName:40s} : {range_str:20s} (range: {varInfo['range']})")

        # Constraints
        print(f"\n📝 CONSTRAINTS ({len(analysis['constraints'])}):")
        print("-" * 80)
        for i, constraint in enumerate(analysis['constraints'][:10]):  # Show first 10
            ctype = constraint['type']
            expr = constraint['expression'][:70]  # Truncate long expressions
            priority = f" [priority={constraint['priority']}]" if 'priority' in constraint else ""
            print(f"  {i+1:2d}. [{ctype:20s}]{priority} {expr}")
        if len(analysis['constraints']) > 10:
            print(f"  ... and {len(analysis['constraints']) - 10} more constraints")

        # Memory constraints
        if analysis['memory_constraints']:
            print(f"\n💾 MEMORY CONSTRAINTS:")
            print("-" * 80)
            for mc in analysis['memory_constraints']:
                print(f"  {mc['memory_level']:15s} (size: {mc['memory_size']:10d} bytes)")
                print(f"    {mc['constraint'][:70]}")

        # Warnings
        if analysis['warnings']:
            print(f"\n⚠️  WARNINGS:")
            print("-" * 80)
            for warning in analysis['warnings']:
                print(f"  {warning}")

        # Suggestions
        if analysis['suggestions']:
            print(f"\n💡 SUGGESTIONS:")
            print("-" * 80)
            for suggestion in analysis['suggestions']:
                print(f"  {suggestion}")

        print("\n" + "=" * 80 + "\n")

    def visualize_tiling_solution(self,
                                  tilingSchedule: TilingSchedule,
                                  variableReplacement: VariableReplacementScheme,
                                  operatorName: str = "Unknown",
                                  tensorShapes: Optional[Dict[str, Tuple[int, ...]]] = None):
        """
        Visualize a tiling solution with text-based graphics.

        Args:
            tilingSchedule: The tiling schedule to visualize
            variableReplacement: Variable replacements for each tile
            operatorName: Name of the operator
            tensorShapes: Full shapes of tensors (for context)
        """
        print("\n" + "=" * 80)
        print(f"  TILING SOLUTION VISUALIZATION: {operatorName}")
        print("=" * 80)

        num_tiles = len(tilingSchedule.inputLoadSchedule)
        print(f"\n📦 Number of tiles: {num_tiles}")

        # Show input/output base addresses
        print(f"\n📍 BASE ADDRESSES:")
        print("-" * 80)
        print("  Input buffers:")
        for name, addrs in tilingSchedule.inputBaseOffsets.items():
            print(f"    {name:20s}: {addrs}")
        print("  Output buffers:")
        for name, addrs in tilingSchedule.outputBaseOffsets.items():
            print(f"    {name:20s}: {addrs}")

        # Show per-tile information
        print(f"\n🔲 TILE DETAILS:")
        print("-" * 80)

        for tileIdx in range(min(num_tiles, 10)):  # Show first 10 tiles
            print(f"\n  Tile {tileIdx}:")

            # Input tiles
            inputSchedule = tilingSchedule.inputLoadSchedule[tileIdx]
            for bufferName, hyperRect in inputSchedule.items():
                print(f"    📥 {bufferName:15s}: offset={hyperRect.offset}, dims={hyperRect.dims}")

            # Output tiles
            outputSchedule = tilingSchedule.outputLoadSchedule[tileIdx]
            for bufferName, hyperRect in outputSchedule.items():
                print(f"    📤 {bufferName:15s}: offset={hyperRect.offset}, dims={hyperRect.dims}")

            # Variable replacements
            if variableReplacement.perTileReplacements:
                print(f"    🔄 Variables:")
                for varName, values in variableReplacement.perTileReplacements.items():
                    if tileIdx < len(values):
                        print(f"       {varName:20s} = {values[tileIdx]}")

        if num_tiles > 10:
            print(f"\n  ... and {num_tiles - 10} more tiles")

        # Try to visualize 2D tiling pattern
        if tensorShapes:
            self._visualize_2d_tiling_pattern(tilingSchedule, tensorShapes)

        print("\n" + "=" * 80 + "\n")

    def _visualize_2d_tiling_pattern(self,
                                     tilingSchedule: TilingSchedule,
                                     tensorShapes: Dict[str, Tuple[int, ...]]):
        """
        Create ASCII art visualization of 2D tiling pattern.

        This works best for Conv2D-like operations with spatial dimensions.
        """
        # Find a tensor with 2D spatial dimensions (NHWC layout)
        target_tensor = None
        target_shape = None

        for name, shape in tensorShapes.items():
            if len(shape) >= 3:  # At least H, W, C
                target_tensor = name
                target_shape = shape
                break

        if not target_tensor:
            return

        print(f"\n🎨 2D TILING PATTERN ({target_tensor}):")
        print("-" * 80)

        # Extract spatial dimensions (assume NHWC: shape[1]=H, shape[2]=W)
        H, W = target_shape[1], target_shape[2]

        # Create a grid
        grid_h = min(H, 40)  # Limit visualization size
        grid_w = min(W, 80)
        scale_h = H / grid_h
        scale_w = W / grid_w

        grid = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]

        # Mark tiles
        tile_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for tileIdx, outputSchedule in enumerate(tilingSchedule.outputLoadSchedule):
            if target_tensor not in outputSchedule:
                continue

            hyperRect = outputSchedule[target_tensor]
            if len(hyperRect.offset) < 3:
                continue

            # Get spatial offset and dims
            h_offset = hyperRect.offset[1]
            w_offset = hyperRect.offset[2]
            h_size = hyperRect.dims[1]
            w_size = hyperRect.dims[2]

            # Map to grid coordinates
            grid_h_start = int(h_offset / scale_h)
            grid_w_start = int(w_offset / scale_w)
            grid_h_end = int((h_offset + h_size) / scale_h)
            grid_w_end = int((w_offset + w_size) / scale_w)

            # Fill grid
            tile_char = tile_chars[tileIdx % len(tile_chars)]
            for gh in range(grid_h_start, min(grid_h_end, grid_h)):
                for gw in range(grid_w_start, min(grid_w_end, grid_w)):
                    if grid[gh][gw] == ' ':
                        grid[gh][gw] = tile_char

        # Print grid
        print(f"  Tensor shape: {target_shape} (H={H}, W={W})")
        print(f"  Grid scale: {scale_h:.2f}x{scale_w:.2f} (tensor units per character)")
        print()
        print("  +" + "-" * grid_w + "+")
        for row in grid:
            print("  |" + "".join(row) + "|")
        print("  +" + "-" * grid_w + "+")
        print()
        print("  Each character represents a region of the tensor.")
        print("  Same character = same tile. Different characters = different tiles.")

    def explain_constraint_failure(self,
                                   error_msg: str,
                                   tilerModel: Optional[TilerModel] = None) -> str:
        """
        Provide user-friendly explanation and suggestions for constraint failures.

        Args:
            error_msg: The error message from the solver
            tilerModel: Optional TilerModel for additional context

        Returns:
            Helpful explanation with suggestions
        """
        explanations = []

        # Parse common error types
        if "No solution found" in error_msg:
            explanations.append("❌ CONSTRAINT SOLVER FAILURE: No valid tiling found")
            explanations.append("")
            explanations.append("This means the constraints are either:")
            explanations.append("  1. Over-constrained (mathematically impossible)")
            explanations.append("  2. Memory is too small for minimum tile size")
            explanations.append("")
            explanations.append("🔍 DEBUGGING STEPS:")
            explanations.append("  1. Increase memory size (e.g., --l1=32000)")
            explanations.append("  2. Check if policy constraints are too strict")
            explanations.append("  3. Look for PerformanceHints that can be relaxed")
            explanations.append("  4. Verify geometrical constraints are correct")

        elif "minimal memory requirement violated" in error_msg:
            # Extract memory level and size from error
            explanations.append("❌ MEMORY SIZE TOO SMALL")
            explanations.append("")
            explanations.append(error_msg)
            explanations.append("")
            explanations.append("🔍 SOLUTIONS:")
            explanations.append("  1. Increase memory size as suggested above")
            explanations.append("  2. Reduce minimum tile size constraints if possible")
            explanations.append("  3. Check if some dimensions can be tiled more aggressively")

        elif "infeasible" in error_msg.lower():
            explanations.append("❌ INFEASIBLE CONSTRAINTS")
            explanations.append("")
            explanations.append("Some constraints cannot be satisfied simultaneously.")
            explanations.append("")
            explanations.append("🔍 DEBUGGING STEPS:")
            explanations.append("  1. Check the 'offending constraints' listed in the error")
            explanations.append("  2. Verify each constraint individually")
            explanations.append("  3. Look for contradictions (e.g., x >= 10 and x <= 5)")

        else:
            explanations.append("❌ TILING ERROR")
            explanations.append("")
            explanations.append(error_msg)

        # Add model-specific analysis if available
        if tilerModel:
            explanations.append("")
            explanations.append("📊 CONSTRAINT MODEL SUMMARY:")
            explanations.append(f"  Variables: {len(tilerModel._variables)}")
            explanations.append(f"  Geometrical constraints: {len(tilerModel._constraints)}")
            explanations.append(f"  Memory constraints: {len(tilerModel._memoryConstraints)}")
            explanations.append(f"  Performance hints: {len(tilerModel._performanceConstraints)}")

        return "\n".join(explanations)

    def compare_tiling_solutions(self,
                                solution1: Tuple[VariableReplacementScheme, TilingSchedule],
                                solution2: Tuple[VariableReplacementScheme, TilingSchedule],
                                names: Tuple[str, str] = ("Solution 1", "Solution 2")):
        """
        Compare two tiling solutions side-by-side.

        Useful for understanding the impact of different constraints or memory sizes.
        """
        print("\n" + "=" * 80)
        print(f"  TILING SOLUTION COMPARISON")
        print("=" * 80)

        var_repl1, sched1 = solution1
        var_repl2, sched2 = solution2

        print(f"\n📊 OVERVIEW:")
        print("-" * 80)
        print(f"  {names[0]:30s} | {names[1]:30s}")
        print(f"  {'-' * 30} | {'-' * 30}")
        print(f"  Tiles: {len(sched1.inputLoadSchedule):24d} | Tiles: {len(sched2.inputLoadSchedule)}")

        # Compare memory usage
        print(f"\n💾 MEMORY USAGE:")
        print("-" * 80)

        # This is a simplified comparison - actual memory usage calculation
        # would require more context
        print(f"  (Detailed memory comparison requires additional context)")

        print("\n" + "=" * 80 + "\n")


def debug_tile_constraint(tileConstraintClass,
                          parseDict: Dict,
                          ctxt: NetworkContext,
                          tilerModel: TilerModel,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to debug a TileConstraint during development.

    Usage:
        from Deeploy.TilingExtension.TileConstraintDebugger import debug_tile_constraint

        # In your TileConstraint's addGeometricalConstraint:
        @staticmethod
        def addGeometricalConstraint(tilerModel, parseDict, ctxt):
            # Add your constraints...
            tilerModel.addConstraint(...)

            # Debug the model
            debug_tile_constraint(
                tileConstraintClass=MyConstraint,
                parseDict=parseDict,
                ctxt=ctxt,
                tilerModel=tilerModel
            )

            return tilerModel

    Args:
        tileConstraintClass: The TileConstraint class being debugged
        parseDict: Operator parse dictionary
        ctxt: Network context
        tilerModel: The TilerModel with constraints added
        verbose: Whether to print detailed output

    Returns:
        Analysis dictionary
    """
    debugger = TileConstraintDebugger(verbose=verbose)
    operator_name = parseDict.get('nodeName', tileConstraintClass.__name__)

    analysis = debugger.analyze_constraint_model(tilerModel, operator_name)

    return analysis
