#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
Example PyTorch models for testing the Deeploy test generator.

These models can be used with create_deeploy_test.py to create test cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinear(nn.Module):
    """
    Simple linear (fully connected) layer.

    Input: (batch_size, input_dim)
    Output: (batch_size, output_dim)
    """

    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleCNN(nn.Module):
    """
    Simple convolutional neural network.

    Input: (batch_size, 3, 32, 32)
    Output: (batch_size, 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleRNN(nn.Module):
    """
    Simple recurrent neural network.

    Input: (batch_size, seq_length, input_dim)
    Output: (batch_size, hidden_dim)
    """

    def __init__(self, input_dim=10, hidden_dim=20, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]  # Return last timestep


class SimpleAttention(nn.Module):
    """
    Simple attention mechanism.

    Input: (batch_size, seq_length, embed_dim)
    Output: (batch_size, seq_length, embed_dim)
    """

    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class SimpleMLP(nn.Module):
    """
    Simple multi-layer perceptron.

    Input: (batch_size, input_dim)
    Output: (batch_size, output_dim)
    """

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleResBlock(nn.Module):
    """
    Simple residual block.

    Input: (batch_size, channels, height, width)
    Output: (batch_size, channels, height, width)
    """

    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class SimpleGEMM(nn.Module):
    """
    Simple General Matrix Multiply (GEMM) operation.
    This is essentially a linear layer without bias.

    Input: (batch_size, input_dim)
    Output: (batch_size, output_dim)
    """

    def __init__(self, input_dim=32, output_dim=16):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


class SimpleDepthwiseConv(nn.Module):
    """
    Simple depthwise convolution.

    Input: (batch_size, channels, height, width)
    Output: (batch_size, channels, height, width)
    """

    def __init__(self, channels=16):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels  # Depthwise
        )

    def forward(self, x):
        return self.depthwise(x)


class SimplePointwise(nn.Module):
    """
    Simple pointwise (1x1) convolution.

    Input: (batch_size, in_channels, height, width)
    Output: (batch_size, out_channels, height, width)
    """

    def __init__(self, in_channels=16, out_channels=32):
        super().__init__()
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.pointwise(x)


class SimpleSoftmax(nn.Module):
    """
    Simple softmax layer.

    Input: (batch_size, num_classes)
    Output: (batch_size, num_classes)
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)


class SimpleElementwise(nn.Module):
    """
    Simple element-wise operations (add, multiply).

    Input: Two tensors of shape (batch_size, channels, height, width)
    Output: (batch_size, channels, height, width)
    """

    def __init__(self, operation='add'):
        super().__init__()
        self.operation = operation

    def forward(self, x, y):
        if self.operation == 'add':
            return x + y
        elif self.operation == 'mul':
            return x * y
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


if __name__ == "__main__":
    """
    Test the models to ensure they work correctly.
    """
    print("Testing example models...")

    # Test SimpleLinear
    model = SimpleLinear(10, 5)
    x = torch.randn(1, 10)
    y = model(x)
    print(f"✓ SimpleLinear: {x.shape} -> {y.shape}")

    # Test SimpleCNN
    model = SimpleCNN(10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"✓ SimpleCNN: {x.shape} -> {y.shape}")

    # Test SimpleMLP
    model = SimpleMLP(784, 128, 10)
    x = torch.randn(1, 784)
    y = model(x)
    print(f"✓ SimpleMLP: {x.shape} -> {y.shape}")

    # Test SimpleGEMM
    model = SimpleGEMM(32, 16)
    x = torch.randn(1, 32)
    y = model(x)
    print(f"✓ SimpleGEMM: {x.shape} -> {y.shape}")

    print("\nAll models tested successfully!")
    print("\nExample usage with create_deeploy_test.py:")
    print("\npython create_deeploy_test.py \\")
    print("    --model-path example_models.py \\")
    print("    --model-class SimpleGEMM \\")
    print("    --test-name testSimpleGEMM \\")
    print("    --input-shapes \"1,32\" \\")
    print("    --seed 42 \\")
    print("    --verbose")
