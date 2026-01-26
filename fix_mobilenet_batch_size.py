#!/usr/bin/env python3
"""Fix MobileNetV1 ONNX model to have batch size of 1."""

import onnx
from onnx import shape_inference
from onnx.tools import update_model_dims

# Load the model
model_path = "DeeployTest/Tests/Models/MobileNetv1/x0_5/network.onnx"
model = onnx.load(model_path)

print("Original model:")
print(f"  Input shape: {[d.dim_value if d.HasField('dim_value') else d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim]}")
print(f"  Output shape: {[d.dim_value if d.HasField('dim_value') else d.dim_param for d in model.graph.output[0].type.tensor_type.shape.dim]}")

# Update batch dimension to 1
updated_model = update_model_dims.update_inputs_outputs_dims(
    model,
    {'input': [1, 3, 224, 224]},
    {'output': [1, 1000]}
)

# Infer shapes to propagate the change
updated_model = shape_inference.infer_shapes(updated_model)

print("\nUpdated model:")
print(f"  Input shape: {[d.dim_value for d in updated_model.graph.input[0].type.tensor_type.shape.dim]}")
print(f"  Output shape: {[d.dim_value for d in updated_model.graph.output[0].type.tensor_type.shape.dim]}")

# Save the updated model
onnx.save(updated_model, model_path)
print(f"\nSaved fixed model to {model_path}")
