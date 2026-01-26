#!/usr/bin/env python3
"""Replace GlobalAveragePool with ReduceMean in MobileNetV1 ONNX model."""

import onnx
from onnx import helper, numpy_helper
import numpy as np

# Load the model
model_path = "DeeployTest/Tests/Models/MobileNetv1/x0_5/network.onnx"
model = onnx.load(model_path)

print(f"Original model: {len([n for n in model.graph.node if n.op_type == 'GlobalAveragePool'])} GlobalAveragePool nodes")

# Find all GlobalAveragePool nodes
gap_nodes = [n for n in model.graph.node if n.op_type == 'GlobalAveragePool']

# Create new nodes list
new_nodes = []
axes_constant_name = None

for node in model.graph.node:
    if node.op_type == 'GlobalAveragePool':
        # For ONNX opset 18, axes is an input (not an attribute)
        # Create a constant for axes if we haven't already
        if axes_constant_name is None:
            axes_constant_name = node.name.replace('GlobalAveragePool', 'axes_const')
            axes_tensor = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), axes_constant_name)
            model.graph.initializer.append(axes_tensor)

        # Replace with ReduceMean
        # GlobalAveragePool on 4D tensor (N, C, H, W) averages over H and W (axes 2 and 3)
        reduce_mean_node = helper.make_node(
            'ReduceMean',
            inputs=[node.input[0], axes_constant_name],
            outputs=node.output,
            name=node.name.replace('GlobalAveragePool', 'ReduceMean'),
            keepdims=1    # Keep the dimensions as 1x1
        )
        new_nodes.append(reduce_mean_node)
        print(f"Replaced {node.name} with {reduce_mean_node.name}")
    else:
        new_nodes.append(node)

# Update the graph
del model.graph.node[:]
model.graph.node.extend(new_nodes)

print(f"\nUpdated model: {len([n for n in model.graph.node if n.op_type == 'ReduceMean'])} ReduceMean nodes")
print(f"Updated model: {len([n for n in model.graph.node if n.op_type == 'GlobalAveragePool'])} GlobalAveragePool nodes")

# Save the updated model
onnx.save(model, model_path)
print(f"\nSaved modified model to {model_path}")
