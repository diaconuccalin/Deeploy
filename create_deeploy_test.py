#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
Deeploy Test Generator

This script creates Deeploy test cases from PyTorch models. It automates the process of:
1. Exporting PyTorch models to ONNX
2. Generating random test inputs
3. Computing expected outputs via forward pass
4. Saving all artifacts in the Deeploy test format

Usage:
    python create_deeploy_test.py \\
        --model-path models/my_model.py \\
        --model-class MyModel \\
        --test-name testMyModel \\
        --input-shapes "1,3,224,224"

See --help for all options.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import checker, shape_inference

# PyTorch imports (with graceful error handling)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Install with: pip install torch")


class DeeployTestGenerator:
    """
    Generator for Deeploy test cases from PyTorch models.

    This class handles:
    - Model loading and validation
    - Input generation
    - ONNX export
    - Output computation
    - Test artifact creation
    """

    def __init__(
        self,
        test_name: str,
        output_dir: str = "DeeployTest/Tests",
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the test generator.

        Args:
            test_name: Name of the test (will be the directory name)
            output_dir: Parent directory for tests (default: DeeployTest/Tests)
            seed: Random seed for reproducibility (default: None)
            verbose: Enable verbose logging
        """
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.test_dir = self.output_dir / test_name
        self.seed = seed
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)
            if PYTORCH_AVAILABLE:
                torch.manual_seed(seed)

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose or level == "ERROR":
            prefix = f"[{level}]"
            print(f"{prefix} {message}")

    def create_test_directory(self) -> Path:
        """
        Create the test directory if it doesn't exist.

        Returns:
            Path to the created test directory
        """
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Created test directory: {self.test_dir}")
        return self.test_dir

    def load_model_from_file(
        self,
        model_path: str,
        model_class: str,
        model_kwargs: Optional[Dict] = None
    ) -> nn.Module:
        """
        Load a PyTorch model from a Python file.

        Args:
            model_path: Path to Python file containing the model
            model_class: Name of the model class
            model_kwargs: Optional kwargs to pass to model constructor

        Returns:
            Instantiated PyTorch model
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Install with: pip install torch")

        # Load the module from file
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module from {model_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["model_module"] = module
        spec.loader.exec_module(module)

        # Get the model class
        if not hasattr(module, model_class):
            raise RuntimeError(f"Class '{model_class}' not found in {model_path}")

        ModelClass = getattr(module, model_class)

        # Instantiate the model
        kwargs = model_kwargs or {}
        model = ModelClass(**kwargs)
        model.eval()

        self.log(f"Loaded model {model_class} from {model_path}")
        return model

    def generate_random_inputs(
        self,
        input_shapes: List[Tuple[int, ...]],
        input_range: Tuple[float, float] = (-1.0, 1.0),
        dtype: np.dtype = np.float32
    ) -> List[np.ndarray]:
        """
        Generate random input arrays.

        Args:
            input_shapes: List of input shapes (each as tuple of ints)
            input_range: Range for random values (min, max)
            dtype: NumPy data type for inputs

        Returns:
            List of random input arrays
        """
        inputs = []
        min_val, max_val = input_range

        for i, shape in enumerate(input_shapes):
            # Generate random values in the specified range
            input_array = np.random.uniform(min_val, max_val, shape).astype(dtype)
            inputs.append(input_array)
            self.log(f"Generated input_{i}: shape={shape}, dtype={dtype}, "
                    f"range=[{input_array.min():.3f}, {input_array.max():.3f}]")

        return inputs

    def run_inference(
        self,
        model: nn.Module,
        inputs: List[np.ndarray],
        capture_activations: bool = False
    ) -> Tuple[List[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Run inference on the model to get expected outputs.

        Args:
            model: PyTorch model
            inputs: List of input arrays
            capture_activations: If True, capture intermediate activations

        Returns:
            Tuple of (outputs, activations)
            - outputs: List of output arrays
            - activations: Dict of intermediate activations (if captured)
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")

        # Convert inputs to torch tensors
        torch_inputs = [torch.from_numpy(inp) for inp in inputs]

        activations = {}

        if capture_activations:
            # Register forward hooks to capture intermediate activations
            def get_activation(name):
                def hook(model, input, output):
                    if isinstance(output, torch.Tensor):
                        activations[name] = output.detach().cpu().numpy()
                return hook

            # Register hooks for all modules
            hooks = []
            for name, module in model.named_modules():
                if name and len(list(module.children())) == 0:  # Leaf modules only
                    hooks.append(module.register_forward_hook(get_activation(name)))

        # Run inference
        with torch.no_grad():
            if len(torch_inputs) == 1:
                output = model(torch_inputs[0])
            else:
                output = model(*torch_inputs)

        # Remove hooks
        if capture_activations:
            for hook in hooks:
                hook.remove()

        # Convert outputs to numpy
        if isinstance(output, torch.Tensor):
            outputs = [output.detach().cpu().numpy()]
        elif isinstance(output, (tuple, list)):
            outputs = [o.detach().cpu().numpy() for o in output]
        else:
            raise RuntimeError(f"Unsupported output type: {type(output)}")

        self.log(f"Computed {len(outputs)} output(s)")
        for i, out in enumerate(outputs):
            self.log(f"  output_{i}: shape={out.shape}, "
                    f"range=[{out.min():.3f}, {out.max():.3f}]")

        return outputs, activations if capture_activations else None

    def export_to_onnx(
        self,
        model: nn.Module,
        inputs: List[np.ndarray],
        opset_version: int = 14,
        simplify: bool = True
    ) -> Path:
        """
        Export PyTorch model to ONNX.

        Args:
            model: PyTorch model
            inputs: Example inputs for tracing
            opset_version: ONNX opset version
            simplify: Whether to simplify the ONNX graph

        Returns:
            Path to the exported ONNX file
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")

        onnx_path = self.test_dir / "network.onnx"

        # Convert inputs to torch tensors
        torch_inputs = [torch.from_numpy(inp) for inp in inputs]
        example_inputs = tuple(torch_inputs) if len(torch_inputs) > 1 else torch_inputs[0]

        # Create input names
        input_names = [f"input_{i}" for i in range(len(inputs))]

        # Export to ONNX
        self.log(f"Exporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            model,
            example_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=[f"output_{i}" for i in range(len(model(example_inputs)
                         if isinstance(model(example_inputs), (tuple, list))
                         else [model(example_inputs)]))],
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=self.verbose
        )

        # Verify the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        checker.check_model(onnx_model)

        # Infer shapes
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, str(onnx_path))

        self.log(f"ONNX model saved to {onnx_path}")

        # Optionally simplify using onnx-simplifier
        if simplify:
            try:
                import onnxsim
                self.log("Simplifying ONNX model...")
                model_simp, check = onnxsim.simplify(str(onnx_path))
                if check:
                    onnx.save(model_simp, str(onnx_path))
                    self.log("ONNX model simplified")
                else:
                    self.log("ONNX simplification check failed, using original model", "WARNING")
            except ImportError:
                self.log("onnx-simplifier not available (optional)", "WARNING")

        return onnx_path

    def save_test_data(
        self,
        inputs: List[np.ndarray],
        outputs: List[np.ndarray],
        activations: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Save test inputs, outputs, and optionally activations as .npz files.

        Args:
            inputs: List of input arrays
            outputs: List of output arrays
            activations: Optional dict of intermediate activations
        """
        # Save inputs
        inputs_path = self.test_dir / "inputs.npz"
        inputs_dict = {f"input_{i}": inp for i, inp in enumerate(inputs)}
        np.savez(str(inputs_path), **inputs_dict)
        self.log(f"Saved {len(inputs)} input(s) to {inputs_path}")

        # Save outputs
        outputs_path = self.test_dir / "outputs.npz"
        outputs_dict = {f"output_{i}": out for i, out in enumerate(outputs)}
        np.savez(str(outputs_path), **outputs_dict)
        self.log(f"Saved {len(outputs)} output(s) to {outputs_path}")

        # Save activations if provided
        if activations:
            activations_path = self.test_dir / "activations.npz"
            np.savez(str(activations_path), **activations)
            self.log(f"Saved {len(activations)} activation(s) to {activations_path}")

    def create_readme(
        self,
        model_info: str,
        input_shapes: List[Tuple[int, ...]],
        additional_info: Optional[str] = None
    ):
        """
        Create a README file for the test.

        Args:
            model_info: Information about the model
            input_shapes: Input shapes used
            additional_info: Any additional information to include
        """
        readme_path = self.test_dir / "README.md"

        content = f"""# {self.test_name}

## Model Information

{model_info}

## Test Details

- **Input shapes**: {', '.join(str(shape) for shape in input_shapes)}
- **Random seed**: {self.seed if self.seed is not None else 'None (random)'}
- **Generated**: Automatically using create_deeploy_test.py

## Files

- `network.onnx`: ONNX model file
- `inputs.npz`: Test input data
- `outputs.npz`: Expected output data
- `activations.npz`: Intermediate activations (if available)

## Running the Test

```bash
cd DeeployTest

# Generic platform
python testRunner_generic.py -t Tests/{self.test_name}

# Siracusa platform
python testRunner_siracusa.py -t Tests/{self.test_name} --cores=8

# With tiling
python testRunner_tiled_siracusa.py -t Tests/{self.test_name} --cores=8 --l1=16000
```

"""

        if additional_info:
            content += f"\n## Additional Information\n\n{additional_info}\n"

        readme_path.write_text(content)
        self.log(f"Created README at {readme_path}")

    def generate_from_pytorch(
        self,
        model: nn.Module,
        input_shapes: List[Tuple[int, ...]],
        input_range: Tuple[float, float] = (-1.0, 1.0),
        capture_activations: bool = False,
        opset_version: int = 14,
        model_info: str = "",
        additional_info: Optional[str] = None
    ):
        """
        Generate a complete Deeploy test from a PyTorch model.

        This is the main entry point that orchestrates the entire process.

        Args:
            model: PyTorch model
            input_shapes: List of input shapes
            input_range: Range for random input values
            capture_activations: Whether to capture intermediate activations
            opset_version: ONNX opset version
            model_info: Description of the model
            additional_info: Additional information for README
        """
        self.log("=" * 60)
        self.log(f"Generating Deeploy test: {self.test_name}")
        self.log("=" * 60)

        # Create test directory
        self.create_test_directory()

        # Generate random inputs
        inputs = self.generate_random_inputs(input_shapes, input_range)

        # Run inference to get outputs
        outputs, activations = self.run_inference(model, inputs, capture_activations)

        # Export to ONNX
        self.export_to_onnx(model, inputs, opset_version)

        # Save test data
        self.save_test_data(inputs, outputs, activations)

        # Create README
        self.create_readme(model_info, input_shapes, additional_info)

        self.log("=" * 60)
        self.log(f"Test generation complete! Test available at: {self.test_dir}")
        self.log("=" * 60)


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    """
    Parse a shape string like "1,3,224,224" into a tuple.

    Args:
        shape_str: Comma-separated shape string

    Returns:
        Tuple of integers representing the shape
    """
    try:
        return tuple(int(x.strip()) for x in shape_str.split(','))
    except ValueError as e:
        raise ValueError(f"Invalid shape string '{shape_str}': {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate Deeploy test cases from PyTorch models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with model file
  python create_deeploy_test.py \\
      --model-path models/my_model.py \\
      --model-class MyModel \\
      --test-name testMyModel \\
      --input-shapes "1,3,224,224"

  # With multiple inputs
  python create_deeploy_test.py \\
      --model-path models/transformer.py \\
      --model-class Transformer \\
      --test-name testTransformer \\
      --input-shapes "1,512" "1,512"

  # With custom range and seed
  python create_deeploy_test.py \\
      --model-path models/my_model.py \\
      --model-class MyModel \\
      --test-name testMyModel \\
      --input-shapes "1,10" \\
      --input-range -5.0 5.0 \\
      --seed 42 \\
      --capture-activations
        """
    )

    # Model specification
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to Python file containing the model class'
    )
    parser.add_argument(
        '--model-class',
        type=str,
        required=True,
        help='Name of the model class to instantiate'
    )
    parser.add_argument(
        '--model-kwargs',
        type=str,
        default=None,
        help='JSON string of kwargs to pass to model constructor (e.g., \'{"num_classes": 10}\')'
    )

    # Test configuration
    parser.add_argument(
        '--test-name',
        type=str,
        required=True,
        help='Name of the test (directory name in DeeployTest/Tests/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='DeeployTest/Tests',
        help='Parent directory for tests (default: DeeployTest/Tests)'
    )

    # Input specification
    parser.add_argument(
        '--input-shapes',
        nargs='+',
        required=True,
        help='Input shapes as comma-separated values (e.g., "1,3,224,224" "1,512")'
    )
    parser.add_argument(
        '--input-range',
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        metavar=('MIN', 'MAX'),
        help='Range for random input values (default: -1.0 1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )

    # ONNX export options
    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX graph simplification'
    )

    # Additional options
    parser.add_argument(
        '--capture-activations',
        action='store_true',
        help='Capture and save intermediate activations'
    )
    parser.add_argument(
        '--model-info',
        type=str,
        default='',
        help='Description of the model for README'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Check PyTorch availability
    if not PYTORCH_AVAILABLE:
        print("ERROR: PyTorch is required but not installed.")
        print("Install with: pip install torch")
        sys.exit(1)

    # Parse input shapes
    try:
        input_shapes = [parse_shape(shape_str) for shape_str in args.input_shapes]
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Parse model kwargs if provided
    model_kwargs = None
    if args.model_kwargs:
        import json
        try:
            model_kwargs = json.loads(args.model_kwargs)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --model-kwargs: {e}")
            sys.exit(1)

    # Create generator
    generator = DeeployTestGenerator(
        test_name=args.test_name,
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=args.verbose
    )

    try:
        # Load model
        model = generator.load_model_from_file(
            args.model_path,
            args.model_class,
            model_kwargs
        )

        # Generate test
        generator.generate_from_pytorch(
            model=model,
            input_shapes=input_shapes,
            input_range=tuple(args.input_range),
            capture_activations=args.capture_activations,
            opset_version=args.opset_version,
            model_info=args.model_info or f"Model: {args.model_class} from {args.model_path}",
            additional_info=None
        )

        print("\n✓ Test generation successful!")
        print(f"\nTo run the test:")
        print(f"  cd DeeployTest")
        print(f"  python testRunner_generic.py -t Tests/{args.test_name}")

    except Exception as e:
        print(f"\nERROR: Test generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
