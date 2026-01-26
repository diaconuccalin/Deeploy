#!/usr/bin/env python3
"""Replace Clip operations with ReLU in MobileNetV1 ONNX model."""

import onnx
from onnx import helper, numpy_helper
import numpy as np

# Load the model
model_path = "DeeployTest/Tests/Models/MobileNetv1/x0_5/network.onnx"
model = onnx.load(model_path)

print(f"Original model: {len([n for n in model.graph.node if n.op_type == 'Clip'])} Clip nodes")

# Find all Clip nodes
clip_nodes = [n for n in model.graph.node if n.op_type == 'Clip']

# Get constant nodes that we'll need to remove
constant_nodes_to_remove = set()
for clip_node in clip_nodes:
    # Clip nodes have 3 inputs: data, min, max
    # The min and max are typically constant nodes
    if len(clip_node.input) >= 2:
        constant_nodes_to_remove.add(clip_node.input[1])  # min constant
    if len(clip_node.input) >= 3:
        constant_nodes_to_remove.add(clip_node.input[2])  # max constant

# Find the constant nodes to remove
nodes_producing_constants = []
for node in model.graph.node:
    if node.op_type == 'Constant' and node.output[0] in constant_nodes_to_remove:
        nodes_producing_constants.append(node)

print(f"Will remove {len(nodes_producing_constants)} Constant nodes")

# Create new nodes list
new_nodes = []
for node in model.graph.node:
    if node.op_type == 'Clip':
        # Replace with ReLU
        # ReLU only takes the input data, not min/max
        relu_node = helper.make_node(
            'Relu',
            inputs=[node.input[0]],  # Only the data input
            outputs=node.output,
            name=node.name.replace('Clip', 'Relu')
        )
        new_nodes.append(relu_node)
        print(f"Replaced {node.name} with {relu_node.name}")
    elif node not in nodes_producing_constants:
        # Keep all other nodes except the constant nodes we're removing
        new_nodes.append(node)

# Update the graph
del model.graph.node[:]
model.graph.node.extend(new_nodes)

print(f"\nUpdated model: {len([n for n in model.graph.node if n.op_type == 'Relu'])} ReLU nodes")
print(f"Updated model: {len([n for n in model.graph.node if n.op_type == 'Clip'])} Clip nodes")

# Save the updated model
onnx.save(model, model_path)
print(f"\nSaved modified model to {model_path}")
