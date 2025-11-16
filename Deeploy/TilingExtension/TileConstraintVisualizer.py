# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
TileConstraintVisualizer: Advanced visualization tools for tiling solutions

This module provides HTML-based interactive visualizations of tiling solutions,
making it easier to understand and debug complex tiling patterns.
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme
from Deeploy.DeeployTypes import NetworkContext


class TileConstraintVisualizer:
    """
    Creates interactive HTML visualizations of tiling solutions.

    Features:
    - 2D heatmaps of tiling patterns
    - Interactive tile information
    - Memory usage charts
    - Constraint dependency graphs
    """

    def __init__(self, output_dir: str = "./tiling_visualizations"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save HTML visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_tiling_schedule(self,
                                  tilingSchedule: TilingSchedule,
                                  variableReplacement: VariableReplacementScheme,
                                  operatorName: str,
                                  tensorShapes: Dict[str, Tuple[int, ...]],
                                  output_file: Optional[str] = None) -> str:
        """
        Create an interactive HTML visualization of a tiling schedule.

        Args:
            tilingSchedule: The tiling schedule to visualize
            variableReplacement: Variable replacements for each tile
            operatorName: Name of the operator
            tensorShapes: Full shapes of all tensors
            output_file: Optional custom output filename

        Returns:
            Path to the generated HTML file
        """
        if output_file is None:
            output_file = f"{operatorName}_tiling.html"

        output_path = self.output_dir / output_file

        html_content = self._generate_html(
            tilingSchedule,
            variableReplacement,
            operatorName,
            tensorShapes
        )

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"✓ Visualization saved to: {output_path}")
        return str(output_path)

    def _generate_html(self,
                      tilingSchedule: TilingSchedule,
                      variableReplacement: VariableReplacementScheme,
                      operatorName: str,
                      tensorShapes: Dict[str, Tuple[int, ...]]) -> str:
        """Generate the HTML content for visualization"""

        num_tiles = len(tilingSchedule.inputLoadSchedule)

        # Generate tile information table
        tile_info_html = self._generate_tile_info_table(
            tilingSchedule,
            variableReplacement,
            num_tiles
        )

        # Generate 2D visualization for spatial tensors
        spatial_viz_html = self._generate_2d_visualization(
            tilingSchedule,
            tensorShapes,
            operatorName
        )

        # Generate memory usage chart
        memory_viz_html = self._generate_memory_usage(
            tilingSchedule,
            tensorShapes
        )

        # Generate statistics
        stats_html = self._generate_statistics(
            tilingSchedule,
            variableReplacement,
            tensorShapes
        )

        # Assemble complete HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tiling Visualization: {operatorName}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .content {{
            padding: 30px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-card .label {{
            font-size: 1em;
            opacity: 0.9;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .tile-grid {{
            font-family: monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre;
            line-height: 1.2;
        }}

        .tile-cell {{
            display: inline-block;
            width: 12px;
            height: 12px;
            margin: 1px;
            border-radius: 2px;
            cursor: pointer;
        }}

        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid #333;
        }}

        .info-box {{
            background: #f0f7ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔲 Tiling Visualization</h1>
            <p>Operator: {operatorName}</p>
        </header>

        <div class="content">
            {stats_html}
            {spatial_viz_html}
            {memory_viz_html}
            {tile_info_html}

            <div class="section">
                <h2>💡 How to Read This Visualization</h2>
                <div class="info-box">
                    <p><strong>Tiles:</strong> The tensor is divided into {num_tiles} tiles that fit in fast memory (L1).</p>
                    <p><strong>2D Grid:</strong> Shows how spatial dimensions are tiled. Same color/character = same tile.</p>
                    <p><strong>Offsets:</strong> Position where the tile starts in the full tensor.</p>
                    <p><strong>Dimensions:</strong> Size of each tile in each dimension.</p>
                </div>
            </div>

            <div class="section">
                <h2>🔧 Debugging Tips</h2>
                <div class="info-box">
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>Large number of tiles may indicate memory constraints are tight</li>
                        <li>Irregular tile sizes may indicate remainder tiles (last tiles in each dimension)</li>
                        <li>Overlapping tiles (for convolution) require extra memory</li>
                        <li>Check variable replacements to see per-tile parameters</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_tile_info_table(self,
                                 tilingSchedule: TilingSchedule,
                                 variableReplacement: VariableReplacementScheme,
                                 num_tiles: int) -> str:
        """Generate HTML table with detailed tile information"""

        rows = []
        for tileIdx in range(num_tiles):
            inputSchedule = tilingSchedule.inputLoadSchedule[tileIdx]
            outputSchedule = tilingSchedule.outputLoadSchedule[tileIdx]

            # Get first input and output for display
            first_input = list(inputSchedule.items())[0] if inputSchedule else (None, None)
            first_output = list(outputSchedule.items())[0] if outputSchedule else (None, None)

            input_info = f"{first_input[1].offset} → {first_input[1].dims}" if first_input[0] else "N/A"
            output_info = f"{first_output[1].offset} → {first_output[1].dims}" if first_output[0] else "N/A"

            # Get variable replacements
            var_info = []
            for varName, values in variableReplacement.perTileReplacements.items():
                if tileIdx < len(values):
                    var_info.append(f"{varName}={values[tileIdx]}")

            var_str = "<br>".join(var_info[:3])  # Show first 3 variables
            if len(var_info) > 3:
                var_str += f"<br>... +{len(var_info) - 3} more"

            rows.append(f"""
                <tr>
                    <td style="text-align: center;"><strong>{tileIdx}</strong></td>
                    <td><code>{input_info}</code></td>
                    <td><code>{output_info}</code></td>
                    <td style="font-size: 0.9em;">{var_str if var_str else "None"}</td>
                </tr>
            """)

        table_html = f"""
            <div class="section">
                <h2>📋 Tile Details</h2>
                <table>
                    <thead>
                        <tr>
                            <th style="width: 80px;">Tile #</th>
                            <th style="width: 30%;">Input (offset → dims)</th>
                            <th style="width: 30%;">Output (offset → dims)</th>
                            <th>Variable Replacements</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
        """

        return table_html

    def _generate_2d_visualization(self,
                                  tilingSchedule: TilingSchedule,
                                  tensorShapes: Dict[str, Tuple[int, ...]],
                                  operatorName: str) -> str:
        """Generate 2D grid visualization of tiling pattern"""

        # Find a tensor with spatial dimensions
        target_tensor = None
        target_shape = None

        for name, shape in tensorShapes.items():
            if len(shape) >= 3:  # At least H, W, C (NHWC layout)
                target_tensor = name
                target_shape = shape
                break

        if not target_tensor or len(target_shape) < 3:
            return """
                <div class="section">
                    <h2>🎨 2D Tiling Pattern</h2>
                    <div class="warning-box">
                        <p>⚠️ No suitable tensor found for 2D visualization. Requires tensors with at least 3 dimensions (e.g., NHWC layout).</p>
                    </div>
                </div>
            """

        H, W = target_shape[1], target_shape[2]

        # Create grid visualization
        grid_h = min(H, 50)
        grid_w = min(W, 100)

        # Generate color palette
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
            '#E63946', '#A8DADC', '#457B9D', '#F4A261', '#2A9D8F'
        ]

        grid_html = f"""
            <div class="section">
                <h2>🎨 2D Tiling Pattern: {target_tensor}</h2>
                <div class="info-box">
                    <p><strong>Tensor shape:</strong> {target_shape} (H={H}, W={W})</p>
                    <p><strong>Grid represents:</strong> Spatial dimensions (Height x Width)</p>
                </div>
                <div style="overflow-x: auto; background: white; padding: 20px; border-radius: 5px;">
        """

        # Create SVG visualization
        cell_size = 10
        svg_width = W * cell_size + 100
        svg_height = H * cell_size + 100

        svg_content = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">'

        # Draw grid and tiles
        for tileIdx, outputSchedule in enumerate(tilingSchedule.outputLoadSchedule):
            if target_tensor not in outputSchedule:
                continue

            hyperRect = outputSchedule[target_tensor]
            if len(hyperRect.offset) < 3:
                continue

            h_offset = hyperRect.offset[1]
            w_offset = hyperRect.offset[2]
            h_size = hyperRect.dims[1]
            w_size = hyperRect.dims[2]

            color = colors[tileIdx % len(colors)]

            # Draw rectangle
            x = w_offset * cell_size + 50
            y = h_offset * cell_size + 50
            width = w_size * cell_size
            height = h_size * cell_size

            svg_content += f'''
                <rect x="{x}" y="{y}" width="{width}" height="{height}"
                      fill="{color}" stroke="#333" stroke-width="2" opacity="0.7">
                    <title>Tile {tileIdx}: offset=({h_offset}, {w_offset}), size=({h_size}, {w_size})</title>
                </rect>
                <text x="{x + width/2}" y="{y + height/2}"
                      text-anchor="middle" dominant-baseline="middle"
                      font-size="12" font-weight="bold" fill="#333">
                    {tileIdx}
                </text>
            '''

        # Add axes labels
        svg_content += f'<text x="25" y="25" font-size="14" font-weight="bold">H (0-{H})</text>'
        svg_content += f'<text x="{svg_width - 80}" y="{svg_height - 25}" font-size="14" font-weight="bold">W (0-{W})</text>'

        svg_content += '</svg>'

        grid_html += svg_content
        grid_html += """
                </div>
            </div>
        """

        return grid_html

    def _generate_memory_usage(self,
                              tilingSchedule: TilingSchedule,
                              tensorShapes: Dict[str, Tuple[int, ...]]) -> str:
        """Generate memory usage visualization"""

        # Calculate memory for each tile (simplified)
        memory_per_tile = []

        for inputSchedule in tilingSchedule.inputLoadSchedule:
            tile_memory = 0
            for bufferName, hyperRect in inputSchedule.items():
                tile_size = 1
                for dim in hyperRect.dims:
                    tile_size *= dim
                tile_memory += tile_size  # Assuming 1 byte per element (simplification)

            memory_per_tile.append(tile_memory)

        max_memory = max(memory_per_tile) if memory_per_tile else 0
        min_memory = min(memory_per_tile) if memory_per_tile else 0
        avg_memory = sum(memory_per_tile) / len(memory_per_tile) if memory_per_tile else 0

        # Create bar chart
        bars_html = ""
        for i, mem in enumerate(memory_per_tile[:20]):  # Show first 20 tiles
            bar_height = (mem / max_memory * 100) if max_memory > 0 else 0
            bars_html += f'''
                <div style="display: inline-block; width: 30px; margin: 0 2px; vertical-align: bottom;">
                    <div style="height: 150px; display: flex; align-items: flex-end;">
                        <div style="width: 100%; height: {bar_height}%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 3px;"
                             title="Tile {i}: {mem} bytes">
                        </div>
                    </div>
                    <div style="text-align: center; font-size: 10px; margin-top: 5px;">{i}</div>
                </div>
            '''

        memory_html = f"""
            <div class="section">
                <h2>💾 Memory Usage Per Tile</h2>
                <div class="info-box">
                    <p><strong>Max memory per tile:</strong> {max_memory:,} bytes</p>
                    <p><strong>Min memory per tile:</strong> {min_memory:,} bytes</p>
                    <p><strong>Average memory per tile:</strong> {avg_memory:,.1f} bytes</p>
                </div>
                <div style="background: white; padding: 20px; border-radius: 5px; overflow-x: auto;">
                    {bars_html}
                </div>
            </div>
        """

        return memory_html

    def _generate_statistics(self,
                           tilingSchedule: TilingSchedule,
                           variableReplacement: VariableReplacementScheme,
                           tensorShapes: Dict[str, Tuple[int, ...]]) -> str:
        """Generate summary statistics"""

        num_tiles = len(tilingSchedule.inputLoadSchedule)
        num_inputs = len(tilingSchedule.inputBaseOffsets)
        num_outputs = len(tilingSchedule.outputBaseOffsets)
        num_vars = len(variableReplacement.perTileReplacements)

        stats_html = f"""
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="value">{num_tiles}</div>
                        <div class="label">Total Tiles</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{num_inputs}</div>
                        <div class="label">Input Buffers</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{num_outputs}</div>
                        <div class="label">Output Buffers</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{num_vars}</div>
                        <div class="label">Variable Replacements</div>
                    </div>
                </div>
            </div>
        """

        return stats_html
