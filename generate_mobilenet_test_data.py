#!/usr/bin/env python3
"""Generate sample input-output npz files for MobileNetV1 testing."""

import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "DeeployTest/Tests/Models/MobileNetv1/x0_5/network.onnx"
session = ort.InferenceSession(model_path)

# Get input/output names and shapes
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

# Create sample input data (batch_size=1, channels=3, height=224, width=224)
# Using random values scaled to typical image range [-1, 1]
np.random.seed(42)  # For reproducibility
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

print(f"Input shape: {input_data.shape}")
print(f"Input dtype: {input_data.dtype}")

# Run inference to get output
outputs = session.run([output_name], {input_name: input_data})
output_data = outputs[0]

print(f"Output shape: {output_data.shape}")
print(f"Output dtype: {output_data.dtype}")

# Save input and output as npz files
output_dir = "DeeployTest/Tests/Models/MobileNetv1/x0_5/"
np.savez(output_dir + "inputs.npz", **{input_name: input_data})
np.savez(output_dir + "outputs.npz", **{output_name: output_data})

print(f"\nSaved {output_dir}inputs.npz")
print(f"Saved {output_dir}outputs.npz")
print(f"\nTest files generated successfully!")
