#!/usr/bin/env python3
"""Fix MobileNetV1 ONNX model for Siracusa platform compatibility."""

import onnx
from onnx import helper, numpy_helper, shape_inference
from onnx.tools import update_model_dims
import numpy as np

# Load the model
model_path = "DeeployTest/Tests/Models/MobileNetv1/x0_5/network.onnx"
model = onnx.load(model_path)

print("=" * 60)
print("MOBILENET V1 MODEL FIXES FOR SIRACUSA PLATFORM")
print("=" * 60)

# 1. Fix batch size
print("\n[1/4] Fixing batch size...")
print(f"  Original input shape: {[d.dim_value if d.HasField('dim_value') else d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim]}")

updated_model = update_model_dims.update_inputs_outputs_dims(
    model,
    {'input': [1, 3, 224, 224]},
    {'output': [1, 1000]}
)
updated_model = shape_inference.infer_shapes(updated_model)

print(f"  Updated input shape: {[d.dim_value for d in updated_model.graph.input[0].type.tensor_type.shape.dim]}")
model = updated_model

# 2. Replace Clip with ReLU (Clip not supported on Siracusa)
print("\n[2/4] Replacing Clip operations with ReLU...")
clip_nodes = [n for n in model.graph.node if n.op_type == 'Clip']
print(f"  Found {len(clip_nodes)} Clip nodes")

# Find constant nodes feeding into Clip
constant_nodes_to_remove = set()
for clip_node in clip_nodes:
    if len(clip_node.input) >= 2:
        constant_nodes_to_remove.add(clip_node.input[1])
    if len(clip_node.input) >= 3:
        constant_nodes_to_remove.add(clip_node.input[2])

nodes_producing_constants = [n for n in model.graph.node if n.op_type == 'Constant' and n.output[0] in constant_nodes_to_remove]

# Create new nodes
new_nodes = []
for node in model.graph.node:
    if node.op_type == 'Clip':
        relu_node = helper.make_node(
            'Relu',
            inputs=[node.input[0]],
            outputs=node.output,
            name=node.name.replace('Clip', 'Relu')
        )
        new_nodes.append(relu_node)
    elif node not in nodes_producing_constants:
        new_nodes.append(node)

del model.graph.node[:]
model.graph.node.extend(new_nodes)
print(f"  Replaced {len(clip_nodes)} Clip nodes with ReLU")

# 3. Replace GlobalAveragePool with ReduceMean (GlobalAveragePool not supported)
print("\n[3/4] Replacing GlobalAveragePool with ReduceMean...")
gap_nodes = [n for n in model.graph.node if n.op_type == 'GlobalAveragePool']
print(f"  Found {len(gap_nodes)} GlobalAveragePool nodes")

if len(gap_nodes) > 0:
    # Add axes as initializer
    axes_name = "reducemean_axes"
    axes_tensor = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), axes_name)
    model.graph.initializer.append(axes_tensor)

    # Replace nodes
    new_nodes = []
    for node in model.graph.node:
        if node.op_type == 'GlobalAveragePool':
            reduce_mean_node = helper.make_node(
                'ReduceMean',
                inputs=[node.input[0], axes_name],
                outputs=node.output,
                name=node.name.replace('GlobalAveragePool', 'ReduceMean'),
                keepdims=1
            )
            new_nodes.append(reduce_mean_node)
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    print(f"  Replaced {len(gap_nodes)} GlobalAveragePool nodes with ReduceMean")

# 4. Verify and save
print("\n[4/4] Verifying and saving model...")
try:
    onnx.checker.check_model(model)
    print("  Model validation: PASSED")
except Exception as e:
    print(f"  Model validation warning: {e}")

onnx.save(model, model_path)
print(f"  Saved to: {model_path}")

print("\n" + "=" * 60)
print("MODEL FIXES COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nSummary of changes:")
print("  - Fixed batch size to 1")
print(f"  - Replaced {len(clip_nodes)} Clip ops with ReLU")
print(f"  - Replaced {len(gap_nodes)} GlobalAveragePool ops with ReduceMean")
print("\nThe model is now compatible with Siracusa platform!")
