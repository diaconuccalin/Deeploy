#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Analyze test networks and extract operator dependencies."""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Set
import onnx
from onnx import TensorProto

# Mapping from ONNX data type enum to string
DTYPE_MAP = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.COMPLEX64: "complex64",
    TensorProto.COMPLEX128: "complex128",
    TensorProto.BFLOAT16: "bfloat16",
}

# Known model tests (full models with multiple operators)
MODEL_TESTS = {
    "WaveFormer", "MobileNetv2", "miniMobileNet", "miniMobileNetv2",
    "simpleCNN", "simpleRegression", "Transformer", "Autoencoder1D",
    "EEGFormer", "CCT", "ICCT", "ICCT_8", "ICCT_ITA", "ICCT_ITA_8",
    "MLPerf", "microLlama", "testFloatDemoTinyViT"
}

def get_dtype_string(dtype_enum: int) -> str:
    """Convert ONNX dtype enum to string."""
    return DTYPE_MAP.get(dtype_enum, f"unknown_{dtype_enum}")

def extract_features_from_node(node) -> Set[str]:
    """Extract special features from an ONNX node."""
    features = set()

    for attr in node.attribute:
        attr_name = attr.name.lower()

        # Convolution features
        if attr_name == 'pads' and any(p > 0 for p in attr.ints):
            features.add("padding")
        if attr_name == 'strides' and any(s > 1 for s in attr.ints):
            features.add("strided")
        if attr_name == 'dilations' and any(d > 1 for d in attr.ints):
            features.add("dilated")
        if attr_name == 'group' and attr.i > 1:
            features.add("grouped")

        # Transpose
        if attr_name == 'transb' and attr.i == 1:
            features.add("transposed_B")
        if attr_name == 'transa' and attr.i == 1:
            features.add("transposed_A")

        # Axis operations
        if attr_name == 'axis' or attr_name == 'axes':
            features.add("axis_operation")

        # Keepdims
        if attr_name == 'keepdims':
            features.add("keepdims")

    # Check for bias input
    if node.op_type in ['Conv', 'Gemm', 'ConvTranspose']:
        if len(node.input) > 2 and node.input[2]:
            features.add("bias")

    return features

def categorize_test(test_name: str, operators: List[str]) -> str:
    """Categorize test based on name and operators."""

    # Model tests
    if test_name in MODEL_TESTS:
        return "model_test"

    # Check for subdirectories (like MLPerf, CCT subdirs)
    if any(parent in MODEL_TESTS for parent in test_name.split('/')):
        return "model_test"

    # Integration tests (multiple operators or complex ops)
    complex_ops = {'Attention', 'LayerNorm', 'Softmax', 'GELU', 'RMSNorm'}
    if len(operators) > 3 or any(op in complex_ops for op in operators):
        return "integration_test"

    # Unit tests (single or few operators)
    return "unit_test"

def analyze_onnx_file(onnx_path: str) -> Dict[str, Any]:
    """Analyze an ONNX file and extract operator information."""
    try:
        model = onnx.load(onnx_path)
        graph = model.graph

        # Extract operators
        operators = []
        all_features = set()

        for node in graph.node:
            op_type = node.op_type
            if op_type not in operators:
                operators.append(op_type)

            # Extract features from this node
            features = extract_features_from_node(node)
            all_features.update(features)

        # Extract data types from inputs and outputs
        data_types = set()

        # Input types
        for input_info in graph.input:
            if input_info.type.tensor_type.elem_type:
                dtype = get_dtype_string(input_info.type.tensor_type.elem_type)
                data_types.add(dtype)

        # Output types
        for output_info in graph.output:
            if output_info.type.tensor_type.elem_type:
                dtype = get_dtype_string(output_info.type.tensor_type.elem_type)
                data_types.add(dtype)

        # Value info types
        for value_info in graph.value_info:
            if value_info.type.tensor_type.elem_type:
                dtype = get_dtype_string(value_info.type.tensor_type.elem_type)
                data_types.add(dtype)

        # Initializer types (weights)
        for init in graph.initializer:
            if init.data_type:
                dtype = get_dtype_string(init.data_type)
                data_types.add(dtype)

        # Determine additional features from test name
        test_name = os.path.basename(os.path.dirname(onnx_path))

        if 'requant' in test_name.lower() or 'rq' in test_name.lower():
            all_features.add("requantization")
        if 'float' in test_name.lower():
            all_features.add("floating_point")
        if 'dw' in test_name.lower():
            all_features.add("depthwise")
        if 'batch' in test_name.lower():
            all_features.add("batched")
        if 'large' in test_name.lower():
            all_features.add("large_tensor")
        if 'grad' in test_name.lower():
            all_features.add("gradient")
        if 'train' in test_name.lower():
            all_features.add("training")

        return {
            "operators": operators,
            "data_types": sorted(list(data_types)),
            "features": sorted(list(all_features)),
            "status": "success"
        }

    except Exception as e:
        return {
            "operators": [],
            "data_types": [],
            "features": [],
            "status": f"error: {str(e)}"
        }

def infer_from_test_name(test_name: str) -> Dict[str, Any]:
    """Infer operator information from test name when ONNX is not available."""

    operators = []
    data_types = []
    features = []

    name_lower = test_name.lower()

    # Infer operators from name
    if 'conv' in name_lower:
        operators.append('Conv')
    if 'matmul' in name_lower:
        operators.append('MatMul')
    if 'gemm' in name_lower:
        operators.append('Gemm')
    if 'add' in name_lower or 'adder' in name_lower:
        operators.append('Add')
    if 'mul' in name_lower:
        operators.append('Mul')
    if 'div' in name_lower:
        operators.append('Div')
    if 'relu' in name_lower:
        operators.append('Relu')
    if 'softmax' in name_lower:
        operators.append('Softmax')
    if 'gelu' in name_lower:
        operators.append('Gelu')
    if 'pad' in name_lower:
        operators.append('Pad')
    if 'pool' in name_lower:
        operators.append('MaxPool' if 'max' in name_lower else 'Pool')
    if 'reshape' in name_lower:
        operators.append('Reshape')
    if 'transpose' in name_lower:
        operators.append('Transpose')
    if 'squeeze' in name_lower:
        operators.append('Squeeze')
    if 'concat' in name_lower:
        operators.append('Concat')
    if 'slice' in name_lower:
        operators.append('Slice')
    if 'layernorm' in name_lower or 'norm' in name_lower:
        operators.append('LayerNormalization')
    if 'reduce' in name_lower:
        if 'sum' in name_lower:
            operators.append('ReduceSum')
        elif 'mean' in name_lower:
            operators.append('ReduceMean')
        else:
            operators.append('Reduce')
    if 'quant' in name_lower and 'dequant' not in name_lower:
        operators.append('QuantizeLinear')
    if 'dequant' in name_lower:
        operators.append('DequantizeLinear')
    if 'attention' in name_lower:
        operators.extend(['MatMul', 'Softmax', 'Add'])
    if 'hardswish' in name_lower:
        operators.append('HardSwish')
    if 'rms' in name_lower:
        operators.append('RMSNorm')
    if 'sgd' in name_lower:
        features.append('training')
    if 'crossentropy' in name_lower:
        operators.append('SoftmaxCrossEntropyLoss')

    # Infer data types
    if 'float' in name_lower:
        data_types.append('float32')
        features.append('floating_point')
    else:
        data_types.append('int8')  # Default assumption

    # Infer features
    if 'rq' in name_lower or 'requant' in name_lower:
        features.append('requantization')
    if 'dw' in name_lower:
        features.append('depthwise')
    if 'bias' in name_lower:
        features.append('bias')
    if 'batch' in name_lower:
        features.append('batched')
    if 'large' in name_lower:
        features.append('large_tensor')
    if 'grad' in name_lower:
        features.append('gradient')
    if 'train' in name_lower:
        features.append('training')
    if 'trans' in name_lower:
        features.append('transposed')
    if 'strided' in name_lower or 'stridded' in name_lower:
        features.append('strided')
    if 'padded' in name_lower:
        features.append('padding')
    if 'zerobias' in name_lower:
        features.append('zero_bias')
    if 'nobias' in name_lower:
        features.append('no_bias')
    if 'unsigned' in name_lower:
        features.append('unsigned_weights')
    if 'pointwise' in name_lower:
        features.append('pointwise')

    return {
        "operators": list(set(operators)),
        "data_types": list(set(data_types)),
        "features": list(set(features)),
        "status": "inferred"
    }

def analyze_all_tests(tests_dir: str) -> Dict[str, Dict[str, Any]]:
    """Analyze all test directories and create mapping."""

    result = {}
    tests_path = Path(tests_dir)

    # Get all test directories
    for test_dir in sorted(tests_path.iterdir()):
        if not test_dir.is_dir():
            continue

        test_name = test_dir.name

        # Check for subdirectories (like CCT, MLPerf, microLlama)
        subdirs = [d for d in test_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Process subdirectories
            for subdir in sorted(subdirs):
                subtest_name = f"{test_name}/{subdir.name}"
                onnx_file = subdir / "network.onnx"

                if onnx_file.exists():
                    analysis = analyze_onnx_file(str(onnx_file))
                else:
                    analysis = infer_from_test_name(subtest_name)

                analysis["category"] = categorize_test(subtest_name, analysis["operators"])
                result[subtest_name] = {
                    "operators": analysis["operators"],
                    "data_types": analysis["data_types"],
                    "features": analysis["features"],
                    "category": analysis["category"]
                }
        else:
            # Process main directory
            onnx_file = test_dir / "network.onnx"

            if onnx_file.exists():
                analysis = analyze_onnx_file(str(onnx_file))
            else:
                analysis = infer_from_test_name(test_name)

            analysis["category"] = categorize_test(test_name, analysis["operators"])
            result[test_name] = {
                "operators": analysis["operators"],
                "data_types": analysis["data_types"],
                "features": analysis["features"],
                "category": analysis["category"]
            }

    return result

def main():
    tests_dir = "/home/user/Deeploy/DeeployTest/Tests"
    output_file = "/home/user/Deeploy/docs/test_operator_mapping.json"

    print(f"Analyzing tests in {tests_dir}...")

    mapping = analyze_all_tests(tests_dir)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"Saved mapping to {output_file}")
    print(f"Total tests analyzed: {len(mapping)}")

    # Print summary statistics
    all_operators = set()
    all_dtypes = set()
    categories = {"unit_test": 0, "integration_test": 0, "model_test": 0}

    for test_name, info in mapping.items():
        all_operators.update(info["operators"])
        all_dtypes.update(info["data_types"])
        categories[info["category"]] += 1

    print(f"\nUnique operators: {len(all_operators)}")
    print(f"Data types: {sorted(all_dtypes)}")
    print(f"\nCategories:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
