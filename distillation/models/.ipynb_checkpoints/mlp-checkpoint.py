import torch
from typing import List, Optional, Tuple

from models.conv1d import Conv1D


class MLPBlock(torch.nn.Module):
    """
    Applies a multi-layer perceptron block to a sequence input.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str,
        norm: str,
        residual: bool = True,
    ):
        super(MLPBlock, self).__init__()

        if residual:
            assert (
                input_size == output_size
            ), "Input and output sizes must match for residual connections."

        self.residual = residual

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(input_size, output_size))

        # Add normalization if specified
        if norm == "batch":
            self.mlp.append(torch.nn.BatchNorm1d(output_size))
        elif norm == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization type: {norm}")

        # Add activation function if specified
        if activation == "relu":
            self.mlp.append(torch.nn.ReLU())
        elif activation == "sigmoid":
            self.mlp.append(torch.nn.Sigmoid())
        elif activation == "none":
            pass
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual multi-layer perceptron block.
        """
        if self.residual:
            return x + self.mlp(x)
        return self.mlp(x)


class ResidualMLP(torch.nn.Module):
    """
    Applies a multi-layer perceptron with residual connections to a sequence input.

    Args:
        input_size (int): Size of network input.
        hidden_num (int): Number of hidden layers
        hidden_size (int): Size of hidden layers
        hidden_activation (str): Activation function to apply after each hidden layer ("relu" or "sigmoid").
        hidden_norm (str): Type of normalization to apply after hidden layers ("batch" or "none").
        output_size (int): Number of output features.
        output_activation (str): Activation function to apply after the output layer ("relu" or "sigmoid").
    """

    def __init__(
        self,
        input_size: int,
        hidden_num: int,
        hidden_size: int,
        hidden_activation: str,
        hidden_norm: str,
        output_size: int,
        output_activation: str,
    ):
        super(ResidualMLP, self).__init__()

        self.hidden = torch.nn.Sequential()
        self.output = torch.nn.Sequential()

        # Process each hidden layer setting as a separate linear layer
        input_dimension = input_size
        for _ in range(hidden_num):
            self.hidden.append(
                MLPBlock(
                    input_size=input_dimension,
                    output_size=hidden_size,
                    activation=hidden_activation,
                    norm=hidden_norm,
                    residual=(input_dimension == hidden_size),
                )
            )
            input_dimension = hidden_size

        # Add the output layer
        self.output.append(torch.nn.Linear(input_dimension, output_size))

        if output_activation == "relu":
            self.output.append(torch.nn.ReLU())
        elif output_activation == "sigmoid":
            self.output.append(torch.nn.Sigmoid())
        elif output_activation == "none":
            self.output.append(torch.nn.Identity())
        else:
            raise ValueError(f"Unknown output activation function: {output_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-layer perceptron.
        """
        for layer in self.hidden:
            x = layer(x)

        return self.output(x)
