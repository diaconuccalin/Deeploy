#!/usr/bin/env python3
"""
Script to compare two specific layers and show their signatures.
"""

import onnx
import sys

def get_tensor_shape(tensor_proto):
    """Extract shape from ONNX tensor."""
    return tuple(dim.dim_value for dim in tensor_proto.type.tensor_type.shape.dim)

def extract_layer_details(onnx_path):
    """Extract detailed layer information."""
    model = onnx.load(onnx_path)
    graph = model.graph

    if len(graph.node) == 0:
        return None

    node = graph.node[0]

    details = {
        'op_type': node.op_type,
        'input_shapes': [get_tensor_shape(inp) for inp in graph.input],
        'output_shapes': [get_tensor_shape(out) for out in graph.output],
        'attributes': {},
        'parameters': {}
    }

    # Get attributes
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            details['attributes'][attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            details['attributes'][attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            details['attributes'][attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.FLOATS:
            details['attributes'][attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            details['attributes'][attr.name] = attr.s.decode('utf-8')

    # Get parameter shapes
    for init in graph.initializer:
        details['parameters'][init.name] = {
            'shape': tuple(init.dims),
            'dtype': init.data_type
        }

    return details

def compare_layers(path1, path2):
    """Compare two layers and show differences."""
    print(f"Comparing:\n  {path1}\n  {path2}\n")

    details1 = extract_layer_details(path1)
    details2 = extract_layer_details(path2)

    print("=" * 80)
    print("LAYER 1:")
    print("=" * 80)
    print(f"Op Type: {details1['op_type']}")
    print(f"Input Shapes: {details1['input_shapes']}")
    print(f"Output Shapes: {details1['output_shapes']}")
    print(f"Attributes: {details1['attributes']}")
    print(f"Parameters:")
    for name, info in details1['parameters'].items():
        print(f"  {name}: {info}")

    print("\n" + "=" * 80)
    print("LAYER 2:")
    print("=" * 80)
    print(f"Op Type: {details2['op_type']}")
    print(f"Input Shapes: {details2['input_shapes']}")
    print(f"Output Shapes: {details2['output_shapes']}")
    print(f"Attributes: {details2['attributes']}")
    print(f"Parameters:")
    for name, info in details2['parameters'].items():
        print(f"  {name}: {info}")

    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)

    identical = True

    if details1['op_type'] != details2['op_type']:
        print(f"✗ Op Type differs: {details1['op_type']} vs {details2['op_type']}")
        identical = False
    else:
        print(f"✓ Op Type matches: {details1['op_type']}")

    if details1['input_shapes'] != details2['input_shapes']:
        print(f"✗ Input Shapes differ: {details1['input_shapes']} vs {details2['input_shapes']}")
        identical = False
    else:
        print(f"✓ Input Shapes match: {details1['input_shapes']}")

    if details1['output_shapes'] != details2['output_shapes']:
        print(f"✗ Output Shapes differ: {details1['output_shapes']} vs {details2['output_shapes']}")
        identical = False
    else:
        print(f"✓ Output Shapes match: {details1['output_shapes']}")

    if details1['attributes'] != details2['attributes']:
        print(f"✗ Attributes differ:")
        print(f"    Layer 1: {details1['attributes']}")
        print(f"    Layer 2: {details2['attributes']}")
        identical = False
    else:
        print(f"✓ Attributes match: {details1['attributes']}")

    # Compare parameter shapes (not values)
    params1_shapes = {k: v['shape'] for k, v in details1['parameters'].items()}
    params2_shapes = {k: v['shape'] for k, v in details2['parameters'].items()}

    if params1_shapes != params2_shapes:
        print(f"✗ Parameter shapes differ:")
        print(f"    Layer 1: {params1_shapes}")
        print(f"    Layer 2: {params2_shapes}")
        identical = False
    else:
        print(f"✓ Parameter shapes match: {params1_shapes}")

    print("\n" + "=" * 80)
    if identical:
        print("RESULT: Layers are IDENTICAL (duplicates)")
    else:
        print("RESULT: Layers are DIFFERENT")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_layers.py <onnx_file1> <onnx_file2>")
        sys.exit(1)

    compare_layers(sys.argv[1], sys.argv[2])
