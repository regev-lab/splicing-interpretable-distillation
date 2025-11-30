import torch
from typing import List, Optional, Tuple


class Conv1D(torch.nn.Module):
    """
    Applies a 1D convolution over a sequence input.

    Args:
        input_channels (int): Number of feature channels in sequence input.
        kernels (list): List of kernels, each specified as a tuple of (number of kernels, kernel size)
        padding (str): Padding type for convolution ("same" or "valid")
        pooling (str): Type of pooling to apply after convolution ("max", "avg", or None)
        pooling_size (int): Size of pooling window
        batch_norm (bool): Whether to apply batch normalization after convolution
        activation (str): Activation function to apply after convolution ("relu", "sigmoid", "elu", or None)
    """

    def __init__(
        self,
        input_channels: int,
        kernels: List[Tuple[int, int]],
        padding: str,
        pooling: Optional[str],
        pooling_size: int,
        batch_norm: bool,
        activation: Optional[str],
    ) -> None:
        super(Conv1D, self).__init__()

        self.convs = torch.nn.ModuleDict()
        self.pools = torch.nn.ModuleDict()

        self.pooling = pooling

        # Process each kernel setting as a separate convolution layer
        total_kernels = 0
        for i, (num_kernels, kernel_size) in enumerate(kernels):
            self.convs[str(i)] = torch.nn.Conv1d(
                in_channels=input_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
            )
            if pooling == "max":
                self.pools[str(i)] = torch.nn.MaxPool1d(
                    kernel_size=pooling_size, ceil_mode=True
                )
            elif pooling == "avg":
                self.pools[str(i)] = torch.nn.AvgPool1d(
                    kernel_size=pooling_size, ceil_mode=True
                )

            total_kernels += num_kernels

        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(total_kernels)
        else:
            self.batch_norm = None

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "elu":
            self.activation = torch.nn.ELU()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, seq_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, total_kernels, seq_length)
        """
        # Apply convolutional layers
        if len(self.convs) == 0:
            # Short-circuit and return nothing if this layer is empty
            return x

        convs = []
        for i in self.convs.keys():
            x_c = self.convs[i](x)
            if self.pooling is not None:
                x_c = self.pools[i](x_c)
            convs.append(x_c)
        x = torch.cat(convs, dim=1)

        # Apply batch normalization
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        # Apply activation function
        if self.activation is not None:
            x = self.activation(x)

        return x
