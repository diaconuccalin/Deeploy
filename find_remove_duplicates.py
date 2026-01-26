#!/usr/bin/env python3
"""
Script to find and remove duplicate layers in MobileNet test directory.
Duplicates are defined as layers with identical:
- Input dimensions
- Output dimensions
- Internal number of parameters
- Hyperparameters
"""

import os
import sys
import onnx
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict
import json

def get_tensor_shape(tensor_proto):
    """Extract shape from ONNX tensor."""
    return tuple(dim.dim_value for dim in tensor_proto.type.tensor_type.shape.dim)

def extract_layer_signature(onnx_path):
    """
    Extract a signature that uniquely identifies a layer configuration.
    Returns a tuple that can be used as a hash key.
    """
    try:
        model = onnx.load(onnx_path)
        graph = model.graph

        if len(graph.node) == 0:
            return None

        node = graph.node[0]  # Assuming single node per file

        # Get layer type
        op_type = node.op_type

        # Get input shapes
        input_shapes = []
        for inp in graph.input:
            input_shapes.append(get_tensor_shape(inp))

        # Get output shapes
        output_shapes = []
        for out in graph.output:
            output_shapes.append(get_tensor_shape(out))

        # Get attributes (hyperparameters)
        attributes = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attributes[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                attributes[attr.name] = tuple(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                attributes[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                attributes[attr.name] = tuple(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                attributes[attr.name] = attr.s.decode('utf-8')

        # Get parameter shapes (weights, biases)
        # Sort by shape only, ignore parameter names (they're just ONNX internal IDs)
        param_shapes = []
        for init in graph.initializer:
            param_shapes.append(tuple(init.dims))

        # Create a hashable signature
        signature = (
            op_type,
            tuple(input_shapes),
            tuple(output_shapes),
            tuple(sorted(attributes.items())),
            tuple(sorted(param_shapes))  # Now only contains shapes, not names
        )

        # Create a human-readable description
        description = {
            'op_type': op_type,
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'attributes': attributes,
            'param_shapes': param_shapes  # Now just a list of shapes
        }

        return signature, description

    except Exception as e:
        print(f"Error processing {onnx_path}: {e}", file=sys.stderr)
        return None

def find_duplicates(base_dir):
    """
    Find all duplicate layers in the directory.
    Returns a dict mapping signatures to list of layer directories.
    """
    signature_to_dirs = defaultdict(list)
    signature_to_desc = {}

    base_path = Path(base_dir)

    # Find all network.onnx files
    for onnx_file in base_path.rglob("network.onnx"):
        layer_dir = onnx_file.parent
        result = extract_layer_signature(str(onnx_file))

        if result is not None:
            signature, description = result
            signature_to_dirs[signature].append(layer_dir)
            signature_to_desc[signature] = description

    return signature_to_dirs, signature_to_desc

def remove_duplicates(signature_to_dirs, base_dir, dry_run=False):
    """
    Remove duplicate directories, keeping only the first one.
    Returns statistics about what was removed.
    """
    stats = defaultdict(lambda: {'total': 0, 'kept': 0, 'removed': 0, 'configs': []})

    for signature, dirs in signature_to_dirs.items():
        layer_type = dirs[0].parent.name

        if len(dirs) > 1:
            # Keep the first one, remove the rest
            kept_dir = dirs[0]
            removed_dirs = dirs[1:]

            stats[layer_type]['total'] += len(dirs)
            stats[layer_type]['kept'] += 1
            stats[layer_type]['removed'] += len(removed_dirs)

            # Store configuration details
            stats[layer_type]['configs'].append({
                'signature': signature,
                'count': len(dirs),
                'kept': kept_dir.name,
                'removed': [d.name for d in removed_dirs]
            })

            # Remove duplicate directories
            for dir_to_remove in removed_dirs:
                if dry_run:
                    print(f"[DRY RUN] Would remove: {dir_to_remove}")
                else:
                    print(f"Removing: {dir_to_remove}")
                    shutil.rmtree(dir_to_remove)
        else:
            # No duplicates for this signature - still track it
            stats[layer_type]['total'] += 1
            stats[layer_type]['kept'] += 1
            stats[layer_type]['configs'].append({
                'signature': signature,
                'count': 1,
                'kept': dirs[0].name,
                'removed': []
            })

    return stats

def create_markdown_table(stats, signature_to_desc, output_file):
    """
    Create a markdown file documenting the layer statistics.
    """
    with open(output_file, 'w') as f:
        f.write("# MobileNet v2 x0.35 Layer Deduplication Report\n\n")
        f.write("This table shows the unique layer configurations and their occurrence counts.\n\n")
        f.write("| Layer Name | Layer Type | Occurrences |\n")
        f.write("|------------|------------|-------------|\n")

        # Collect all configurations with their counts
        all_configs = []
        for layer_type in sorted(stats.keys()):
            layer_stats = stats[layer_type]
            for config in layer_stats['configs']:
                all_configs.append({
                    'name': config['kept'],
                    'type': layer_type,
                    'count': config['count']
                })

        # Sort by layer type, then by name
        all_configs.sort(key=lambda x: (x['type'], x['name']))

        for config in all_configs:
            f.write(f"| {config['name']} | {config['type']} | {config['count']} |\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_remove_duplicates.py <layers_directory> [--dry-run]")
        sys.exit(1)

    base_dir = sys.argv[1]
    dry_run = '--dry-run' in sys.argv

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)

    print(f"Scanning directory: {base_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE (will delete files)'}")
    print()

    # Find all layers and their signatures
    print("Finding all layers and computing signatures...")
    signature_to_dirs, signature_to_desc = find_duplicates(base_dir)

    print(f"Found {len(signature_to_dirs)} unique layer configurations")
    print()

    # Find duplicates
    duplicates = {sig: dirs for sig, dirs in signature_to_dirs.items() if len(dirs) > 1}
    print(f"Found {len(duplicates)} configurations with duplicates")
    print()

    # Remove duplicates
    print("Processing duplicates...")
    stats = remove_duplicates(signature_to_dirs, base_dir, dry_run)
    print()

    # Create markdown report
    output_file = os.path.join(base_dir, "layer_deduplication_report.md")
    print(f"Creating report: {output_file}")
    create_markdown_table(stats, signature_to_desc, output_file)

    print()
    print("Summary:")
    for layer_type in sorted(stats.keys()):
        layer_stats = stats[layer_type]
        print(f"  {layer_type}: {layer_stats['total']} total, "
              f"{layer_stats['kept']} kept, {layer_stats['removed']} removed")

    print()
    print(f"Report saved to: {output_file}")

if __name__ == "__main__":
    main()
