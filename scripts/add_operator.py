#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
Add Operator Script - Streamline adding new operators to Deeploy platforms

This script generates all the necessary boilerplate code for adding a new operator
to a specific platform with a specific data type. It creates:
- Parser class
- TypeChecker class
- Template with Mako code
- Binding definition
- Layer class
- Updates to Platform.py mapping

Usage:
    python add_operator.py --operator Conv2D --platform Siracusa --dtype float32
    python add_operator.py -o MatMul -p Generic -d int8
    python add_operator.py --operator Add --platform MemPool --dtype int16 --interactive
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class OperatorGenerator:
    """Generator for new operator implementations."""

    DTYPE_MAPPING = {
        'int8': 'int8_t',
        'int16': 'int16_t',
        'int32': 'int32_t',
        'uint8': 'uint8_t',
        'uint16': 'uint16_t',
        'uint32': 'uint32_t',
        'float16': 'float16_t',
        'float32': 'float32_t',
        'bfloat16': 'bfloat16_t',
    }

    def __init__(self, operator: str, platform: str, dtype: str, interactive: bool = False):
        """Initialize the generator.

        Args:
            operator: Operator name (e.g., Conv2D, MatMul, Add)
            platform: Platform name (e.g., Generic, Siracusa, MemPool)
            dtype: Data type (e.g., int8, float32)
            interactive: Whether to prompt for additional info
        """
        self.operator = operator
        self.platform = platform
        self.dtype = dtype
        self.interactive = interactive

        # Get repository root
        self.repo_root = Path(__file__).parent.parent.absolute()
        self.target_path = self.repo_root / "Deeploy" / "Targets" / platform

        # Validate inputs
        self._validate_inputs()

        # Get dtype internal representation
        self.dtype_internal = self.DTYPE_MAPPING.get(dtype.lower(), dtype + '_t')

        # Generate class names
        self.parser_class = f"{operator}Parser"
        self.checker_class = f"{operator}Checker"
        self.layer_class = f"{operator}Layer"
        self.template_var = f"{operator.lower()}Template"
        self.mapper_var = f"{operator}Mapper"
        self.binding_var = f"{operator}Bindings"

    def _validate_inputs(self):
        """Validate input parameters."""
        if not self.target_path.exists():
            print(f"Error: Platform '{self.platform}' does not exist at {self.target_path}")
            print(f"Available platforms:")
            targets_dir = self.repo_root / "Deeploy" / "Targets"
            for p in sorted(targets_dir.iterdir()):
                if p.is_dir() and not p.name.startswith('_'):
                    print(f"  - {p.name}")
            sys.exit(1)

        if self.dtype.lower() not in self.DTYPE_MAPPING:
            print(f"Warning: Data type '{self.dtype}' is not in standard types.")
            print(f"Standard types: {', '.join(self.DTYPE_MAPPING.keys())}")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != 'y':
                sys.exit(1)

    def generate_spdx_header(self) -> str:
        """Generate SPDX license header."""
        return """# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""

    def generate_parser(self) -> str:
        """Generate Parser class code."""
        return f'''{self.generate_spdx_header()}
from typing import Dict, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeParser, OperatorRepresentation


class {self.parser_class}(NodeParser):
    """Parser for {self.operator} operator."""

    def parseNode(self, node) -> Dict:
        """Parse ONNX node attributes for {self.operator}.

        Args:
            node: ONNX node protobuf

        Returns:
            Dictionary of parsed attributes

        TODO: Extract operator-specific attributes from node
        Example attributes to extract:
            - For Conv: kernel_shape, strides, pads, dilations, group
            - For MatMul: transA, transB
            - For Add: broadcasting parameters
        """
        # TODO: Implement attribute parsing
        attrs = {{}}

        # Example: Extract kernel shape for Conv2D
        # if hasattr(node, 'attribute'):
        #     for attr in node.attribute:
        #         if attr.name == 'kernel_shape':
        #             attrs['kernel_shape'] = list(attr.ints)
        #         elif attr.name == 'strides':
        #             attrs['strides'] = list(attr.ints)

        return attrs

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node,
                      channels_first: bool = True) -> OperatorRepresentation:
        """Parse {self.operator} node with network context.

        Args:
            ctxt: Network context for tensor lookup
            node: ONNX node protobuf
            channels_first: Whether to use channels-first layout

        Returns:
            OperatorRepresentation with inputs, outputs, and attributes
        """
        # Get inputs and outputs from context
        inputs = [ctxt.lookup(inp) for inp in node.input]
        outputs = [ctxt.lookup(out) for out in node.output]

        # Parse attributes
        attrs = self.parseNode(node)

        # TODO: Add operator-specific attribute processing
        # Example: Calculate output shape, padding, etc.

        # Create operator representation
        return OperatorRepresentation(
            '{self.operator}',
            node.name,
            inputs,
            outputs,
            attrs
        )
'''

    def generate_typechecker(self) -> str:
        """Generate TypeChecker class code."""
        return f'''{self.generate_spdx_header()}
from Deeploy.DeeployTypes import NodeTypeChecker, OperatorRepresentation
from Deeploy.CommonExtensions.DataTypes import {self.dtype_internal}


class {self.checker_class}(NodeTypeChecker):
    """Type checker for {self.operator} operator with {self.dtype} precision."""

    def typeCheck(self, operatorRepresentation: OperatorRepresentation) -> bool:
        """Validate type compatibility for {self.operator}.

        Args:
            operatorRepresentation: Operator representation with inputs/outputs

        Returns:
            True if types are valid, False otherwise

        TODO: Implement type checking logic
        - Verify input tensor types match expected dtype
        - Verify output tensor types are correct
        - Check tensor shapes are compatible
        - Validate operator-specific type constraints
        """
        inputs = operatorRepresentation.inputs
        outputs = operatorRepresentation.outputs

        # TODO: Implement type checking
        # Example for element-wise operations:
        # if len(inputs) != 2 or len(outputs) != 1:
        #     return False
        # if inputs[0].dtype != {self.dtype_internal}:
        #     return False
        # if inputs[1].dtype != {self.dtype_internal}:
        #     return False
        # if outputs[0].dtype != {self.dtype_internal}:
        #     return False

        # Placeholder: Accept all types for now
        return True
'''

    def generate_template(self) -> str:
        """Generate Template code."""
        return f'''{self.generate_spdx_header()}
from Deeploy.DeeployTypes import NodeTemplate


# TODO: Implement {self.operator} kernel template
# This template generates C code for the {self.operator} operation
# with {self.dtype} data type.
#
# Available template variables (from OperatorRepresentation):
# - ${{name}}: Node name
# - ${{data_in}}: Input tensor name(s)
# - ${{data_out}}: Output tensor name
# - ${{...}}: Any attributes from parseNode()
#
# Template syntax:
# - ${{variable}}: Insert variable value
# - % for line in range(n): ... % endfor: Python control flow
# - <%...%>: Inline Python code blocks

{self.template_var} = NodeTemplate("""
// TODO: Implement {self.operator} kernel for {self.dtype}

void ${{name}}(
    // TODO: Define function signature
    // Example:
    // {self.dtype_internal} *data_in,
    // {self.dtype_internal} *data_out,
    // int size
) {{
    // TODO: Implement kernel
    // Example for element-wise operation:
    // for (int i = 0; i < size; i++) {{
    //     data_out[i] = compute(data_in[i]);
    // }}
}}
""")


# Optional: Add tiled version for memory-constrained platforms
{self.template_var}Tiled = NodeTemplate("""
// TODO: Implement tiled {self.operator} kernel for {self.dtype}
// This version should work with tiled data and support DMA operations

void ${{name}}_tiled(
    // TODO: Define tiled function signature
) {{
    // TODO: Implement tiled kernel with DMA support
}}
""")
'''

    def generate_binding(self) -> str:
        """Generate Binding code."""
        return f'''{self.generate_spdx_header()}
from Deeploy.DeeployTypes import NodeBinding, NodeMapper
from Deeploy.CommonExtensions.DataTypes import {self.dtype_internal}

# Import generated classes
from .Parsers import {self.parser_class}
from .TypeCheckers import {self.checker_class}
from .Templates.{self.operator}Template import {self.template_var}


# Create binding for {self.operator} with {self.dtype} data type
{self.binding_var} = [
    NodeBinding(
        parser={self.parser_class}(),
        checker={self.checker_class}(),
        template={self.template_var},
        # Signature constraint: define input/output types
        # TODO: Update signature constraint based on your operator
        # Examples:
        # - Element-wise binary: ([{self.dtype_internal}, {self.dtype_internal}], [{self.dtype_internal}])
        # - Unary: ([{self.dtype_internal}], [{self.dtype_internal}])
        # - MatMul: ([{self.dtype_internal}, {self.dtype_internal}], [{self.dtype_internal}])
        signatureConstraint=([{self.dtype_internal}], [{self.dtype_internal}])
    )
]


# Create mapper combining parser, checker, and bindings
{self.mapper_var} = NodeMapper(
    {self.parser_class}(),
    {self.checker_class}(),
    {self.binding_var}
)
'''

    def generate_layer(self) -> str:
        """Generate Layer class code."""
        return f'''{self.generate_spdx_header()}
from typing import List

from Deeploy.DeeployTypes import NodeMapper, ONNXLayer, OperatorRepresentation


class {self.layer_class}(ONNXLayer):
    """Layer implementation for {self.operator} operator."""

    def __init__(self, mappers: List[NodeMapper]):
        """Initialize {self.operator} layer.

        Args:
            mappers: List of NodeMapper instances for different type signatures
        """
        super().__init__('{self.operator}', mappers)

    def computeOps(self, operatorRepresentation: OperatorRepresentation) -> int:
        """Compute operation count for {self.operator}.

        Args:
            operatorRepresentation: Operator representation

        Returns:
            Number of operations (MACs, FLOPs, etc.)

        TODO: Implement operation counting
        This is used for profiling and performance analysis.

        Examples:
        - Conv2D: out_h * out_w * out_c * kernel_h * kernel_w * in_c
        - MatMul: M * N * K
        - Add: size (number of elements)
        """
        # TODO: Implement operation counting
        # Placeholder: return output tensor size
        if operatorRepresentation.outputs:
            output = operatorRepresentation.outputs[0]
            if hasattr(output, 'shape'):
                import numpy as np
                return int(np.prod(output.shape))
        return 0
'''

    def generate_platform_update(self) -> str:
        """Generate code to add to Platform.py."""
        return f'''
# Add to imports section:
from .Layers import {self.layer_class}
from .Bindings import {self.mapper_var}

# Add to {self.platform}Mapping dictionary:
    '{self.operator}': {self.layer_class}([{self.mapper_var}]),
'''

    def create_template_file(self):
        """Create template file in Templates/ directory."""
        templates_dir = self.target_path / "Templates"
        templates_dir.mkdir(exist_ok=True)

        template_file = templates_dir / f"{self.operator}Template.py"
        if template_file.exists():
            print(f"Warning: Template file already exists: {template_file}")
            response = input("Overwrite? [y/N] ")
            if response.lower() != 'y':
                print(f"Skipping template file creation.")
                return

        with open(template_file, 'w') as f:
            f.write(self.generate_template())
        print(f"Created: {template_file}")

        # Update __init__.py in Templates/
        init_file = templates_dir / "__init__.py"
        if init_file.exists():
            with open(init_file, 'r') as f:
                content = f.read()
            if f"{self.operator}Template" not in content:
                # Add import
                import_line = f"\nfrom .{self.operator}Template import {self.template_var}"
                with open(init_file, 'a') as f:
                    f.write(import_line)
                print(f"Updated: {init_file}")

    def update_or_create_file(self, filename: str, new_class: str, content_generator):
        """Update existing file or create new one.

        Args:
            filename: Target filename (e.g., 'Parsers.py')
            new_class: Class name to check for
            content_generator: Function that returns the code to add
        """
        filepath = self.target_path / filename

        if filepath.exists():
            # Check if class already exists
            with open(filepath, 'r') as f:
                existing_content = f.read()

            if f"class {new_class}" in existing_content:
                print(f"Warning: {new_class} already exists in {filepath}")
                response = input("Overwrite? [y/N] ")
                if response.lower() != 'y':
                    print(f"Skipping {filename}")
                    return
                # Remove old class definition
                # This is complex, so we'll just warn the user
                print(f"Please manually remove the old {new_class} definition")
                return

            # Append to existing file
            with open(filepath, 'a') as f:
                f.write('\n\n' + content_generator())
            print(f"Updated: {filepath}")
        else:
            # Create new file
            with open(filepath, 'w') as f:
                f.write(content_generator())
            print(f"Created: {filepath}")

    def update_platform_mapping(self):
        """Update Platform.py with new operator mapping."""
        platform_file = self.target_path / "Platform.py"

        if not platform_file.exists():
            print(f"Error: Platform.py not found at {platform_file}")
            return

        print(f"\n{'='*70}")
        print(f"Manual step required: Update {platform_file}")
        print(f"{'='*70}")
        print(self.generate_platform_update())
        print(f"{'='*70}\n")

    def generate_test_template(self) -> str:
        """Generate test template for the new operator."""
        return f'''{self.generate_spdx_header()}
"""
Test template for {self.operator} operator on {self.platform} platform.

To create a test:
1. Create a directory: DeeployTest/Tests/{self.operator}Test/
2. Create ONNX model: network.onnx
3. Create input data: input.npy
4. Run: python testRunner_{self.platform.lower()}.py -t Tests/{self.operator}Test
"""

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_test_network():
    """Create ONNX test network for {self.operator}."""
    # TODO: Define input/output shapes
    input_shape = [1, 3, 32, 32]  # Example: NCHW format
    output_shape = [1, 3, 32, 32]  # Adjust based on operator

    # Create input/output tensors
    input_tensor = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,  # TODO: Adjust based on dtype
        input_shape
    )
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        output_shape
    )

    # TODO: Create operator node with appropriate attributes
    node = helper.make_node(
        '{self.operator}',
        inputs=['input'],
        outputs=['output'],
        # TODO: Add operator-specific attributes
        # Example for Conv2D:
        # kernel_shape=[3, 3],
        # strides=[1, 1],
        # pads=[1, 1, 1, 1],
    )

    # Create graph
    graph = helper.make_graph(
        [node],
        '{self.operator}Test',
        [input_tensor],
        [output_tensor]
    )

    # Create model
    model = helper.make_model(graph, producer_name='deeploy-test')

    return model


def create_test_input():
    """Create test input data."""
    # TODO: Adjust shape based on your operator
    input_shape = [1, 3, 32, 32]

    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Or create specific test patterns
    # input_data = np.ones(input_shape, dtype=np.float32)

    return input_data


if __name__ == '__main__':
    # Create test network
    model = create_test_network()
    onnx.save(model, 'DeeployTest/Tests/{self.operator}Test/network.onnx')
    print(f"Created network.onnx")

    # Create test input
    input_data = create_test_input()
    np.save('DeeployTest/Tests/{self.operator}Test/input.npy', input_data)
    print(f"Created input.npy")

    print(f"\\nTo run test:")
    print(f"cd DeeployTest")
    print(f"python testRunner_{self.platform.lower()}.py -t Tests/{self.operator}Test")
'''

    def generate_all(self):
        """Generate all files for the new operator."""
        print(f"\n{'='*70}")
        print(f"Generating {self.operator} operator for {self.platform} platform ({self.dtype})")
        print(f"{'='*70}\n")

        # Create Templates/ file
        self.create_template_file()

        # Update or create Parser
        self.update_or_create_file(
            'Parsers.py',
            self.parser_class,
            self.generate_parser
        )

        # Update or create TypeChecker
        self.update_or_create_file(
            'TypeCheckers.py',
            self.checker_class,
            self.generate_typechecker
        )

        # Update or create Binding
        self.update_or_create_file(
            'Bindings.py',
            self.binding_var,
            self.generate_binding
        )

        # Update or create Layer
        self.update_or_create_file(
            'Layers.py',
            self.layer_class,
            self.generate_layer
        )

        # Create test template
        test_template_file = self.repo_root / "scripts" / f"test_template_{self.operator}.py"
        with open(test_template_file, 'w') as f:
            f.write(self.generate_test_template())
        print(f"Created test template: {test_template_file}")

        # Update Platform.py (manual step)
        self.update_platform_mapping()

        print(f"\n{'='*70}")
        print(f"Generation complete!")
        print(f"{'='*70}")
        print(f"\nNext steps:")
        print(f"1. Review generated files in: {self.target_path}")
        print(f"2. Fill in TODOs in each file:")
        print(f"   - Parser: Extract operator attributes")
        print(f"   - TypeChecker: Validate input/output types")
        print(f"   - Template: Implement C kernel")
        print(f"   - Layer: Implement operation counting")
        print(f"3. Update Platform.py as shown above")
        print(f"4. Create test using: python scripts/test_template_{self.operator}.py")
        print(f"5. Run test: cd DeeployTest && python testRunner_{self.platform.lower()}.py -t Tests/{self.operator}Test")
        print(f"6. Format code: make format")
        print(f"7. Run linting: make lint")
        print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate boilerplate code for adding a new operator to Deeploy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add FP32 Conv2D to Siracusa platform
  python add_operator.py --operator Conv2D --platform Siracusa --dtype float32

  # Add int8 MatMul to Generic platform
  python add_operator.py -o MatMul -p Generic -d int8

  # Add operator with interactive prompts
  python add_operator.py -o Add -p MemPool -d int16 --interactive

Available data types:
  int8, int16, int32, uint8, uint16, uint32, float16, float32, bfloat16
        """
    )

    parser.add_argument(
        '-o', '--operator',
        required=True,
        help='Operator name (e.g., Conv2D, MatMul, Add)'
    )
    parser.add_argument(
        '-p', '--platform',
        required=True,
        help='Platform name (e.g., Generic, Siracusa, MemPool)'
    )
    parser.add_argument(
        '-d', '--dtype',
        required=True,
        help='Data type (e.g., int8, float32, float16)'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Enable interactive mode for additional prompts'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without creating files'
    )

    args = parser.parse_args()

    try:
        generator = OperatorGenerator(
            args.operator,
            args.platform,
            args.dtype,
            args.interactive
        )

        if args.dry_run:
            print("DRY RUN MODE - No files will be created")
            print(f"\nWould generate:")
            print(f"  - {generator.target_path / 'Templates' / f'{args.operator}Template.py'}")
            print(f"  - Updates to {generator.target_path / 'Parsers.py'}")
            print(f"  - Updates to {generator.target_path / 'TypeCheckers.py'}")
            print(f"  - Updates to {generator.target_path / 'Bindings.py'}")
            print(f"  - Updates to {generator.target_path / 'Layers.py'}")
            print(f"  - Test template in scripts/")
        else:
            generator.generate_all()

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
