import math
import torch
from typing import List, Optional, Tuple

from models.positional.index import IndexPositionalEncoding
from models.positional.sinusoidal import (
    SineCosinePositionalEncoding,
    HybridSineCosinePositionalEncoding,
)
from models.conv1d import Conv1D
from models.mlp import ResidualMLP, MLPBlock
from utils import utils


class FeatureBasesNN(torch.nn.Module):
    """
    Applies a multi-layer perceptron to the input using the Neural Basis Model described in:
    https://arxiv.org/abs/2205.14120

    Args:
        num_features (int): Number of input features (i.e. number of convolutional kernels in the input)
        feature_size (int): The size of each input feature
        order (int): Order of feature interactions to compute as terms
        num_bases (int): Number of shared base functions to learn
        feature_hidden_num (int): Number of hidden layers in the MLP
        feature_hidden_size (int): Size of the hidden layers in the MLP
        feature_activation (str): Activation function for the hidden layers in the MLP
        feature_norm (str): Normalization layer to apply after each hidden layer in the MLP
        per_feature_top_k (int): Number of positions per feature to select (can be None to use all) for forming terms
    """

    def __init__(
        self,
        num_features: int,
        feature_size: int,
        order: int,
        num_bases: int,
        feature_hidden_num: int,
        feature_hidden_size: int,
        feature_activation: str,
        feature_norm: str,
        per_feature_top_k: Optional[int] = None,
    ):
        super(FeatureBasesNN, self).__init__()

        self.num_features = num_features
        self.feature_size = feature_size
        self.order = order
        self.num_bases = num_bases

        self.feature_hidden_num = feature_hidden_num
        self.feature_hidden_size = feature_hidden_size
        self.feature_activation = feature_activation
        self.feature_norm = feature_norm

        self.per_feature_top_k = per_feature_top_k

        self.base_nn = torch.nn.Sequential()

        input_dim = feature_size * order
        for i in range(feature_hidden_num):
            self.base_nn.append(
                MLPBlock(
                    input_size=input_dim,
                    output_size=feature_hidden_size,
                    activation=feature_activation,
                    norm=feature_norm,
                    residual=(input_dim == feature_hidden_size),
                )
            )
            input_dim = feature_hidden_size

        # Each base function is one output of the MLP
        self.base_nn.append(
            MLPBlock(
                input_size=input_dim,
                output_size=num_bases,
                activation=feature_activation,
                norm=feature_norm,
                residual=False,  # No residual connection for the final layer
            )
        )

        # Shape function network
        self.shape_fn_nn = torch.nn.Sequential()
        num_shape_fns = utils.num_combinations_with_replacement(num_features, order)
        self.shape_fn_nn.append(
            torch.nn.Conv1d(
                in_channels=num_shape_fns * num_bases,
                out_channels=num_shape_fns,
                kernel_size=1,
                groups=num_shape_fns,
            ),
        )

        # Precalculate feature combinations (since they are fixed for a given order and expensive to re-do every loop)
        feature_combinations = torch.combinations(
            torch.arange(num_features),
            r=self.order,
            with_replacement=True,
        )
        self.register_buffer("feature_combinations", feature_combinations)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the neural basis model. The output tensor has shape
        (batch_size, ncr(num_features, order), per_feature_top_k**order) or
        (batch_size, ncr(num_features, order), sequence_length**order) (if per_feature_top_k is None)

        where ncr(n, r) is the number of combinations with replacement of r elements from a set of n elements.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, feature_size, sequence_length).
        """
        batch_size, num_features, feature_size, sequence_length = x.size()

        assert num_features == self.num_features
        assert feature_size == self.feature_size

        if self.per_feature_top_k is None:
            # Use all positions during evaluation or when top_k is not set
            num_positions = sequence_length
        else:
            # Subset x to the top_k positions for each feature
            num_positions = self.per_feature_top_k
            _, top_k_idx_per_feat = torch.topk(
                x[:, :, 0, :].reshape(batch_size * num_features, -1),
                self.per_feature_top_k,
                dim=1,
                sorted=False,
            )

            x = torch.gather(
                x,
                dim=3,
                index=top_k_idx_per_feat.reshape(
                    batch_size, num_features, 1, self.per_feature_top_k
                ).expand(
                    -1,
                    -1,
                    feature_size,
                    -1,
                ),
            )

        # Number of combinations of features of order `order` with replacement
        num_combos = utils.num_combinations_with_replacement(num_features, self.order)

        # Calculate all position combinations
        pos_tensor = torch.arange(num_positions, device=x.device)
        pos_combinations = torch.cartesian_prod(
            *[pos_tensor for _ in range(self.order)]
        )

        # Create indices for feature and position combinations
        feature_indices = (
            self.feature_combinations.unsqueeze(1)
            .expand(-1, num_positions**self.order, -1)
            .reshape(-1, self.order)
            .flatten()
        )
        pos_indices = pos_combinations.repeat(
            self.feature_combinations.size(0), 1
        ).flatten()

        # Flatten the input tensor to have shape (batch_size, num_features * num_positions, feature_size)
        # so that we can index into it more easily
        x_features_flattened = x.permute(0, 1, 3, 2).reshape(
            batch_size, num_features * num_positions, feature_size
        )

        # Calculate all position combinations (for now, use Cartesian product, since
        # deduplication is a bit tricky when making the feature tensor as it depends
        # on whether the chosen features have duplicates).
        x_features_flattened_idxs = feature_indices * num_positions + pos_indices

        # x_in is a (batch_size, num_positions ** order, num_feature_combinations, feature_size * order) tensor
        # where x_in[a, b, c, :] = In batch a, the value of feature combination with rank c
        # and position combination with rank b
        #
        # Currently, indexing this is quite expensive. We can probably do something smarter in the
        # earlier stages so that we get better cache hits.
        x_in = (
            torch.index_select(
                x_features_flattened, dim=1, index=x_features_flattened_idxs
            )
            .reshape(
                batch_size,
                num_combos,
                num_positions**self.order,
                feature_size * self.order,
            )
            .permute(0, 2, 1, 3)
        )

        # Apply the base network to the input tensor
        x = (
            self.base_nn(x_in.reshape(-1, feature_size * self.order))
            .reshape(batch_size, num_positions**self.order, self.num_bases * num_combos)
            .permute(0, 2, 1)
        )

        # Apply the shape function network to the output of the base network
        for layer in self.shape_fn_nn:
            # BatchNorm is applied across the channel dimension, so we need
            # to reshape to keep the features separate
            if isinstance(layer, torch.nn.BatchNorm1d):
                x_size = x.size()
                x = layer(x.reshape(batch_size, -1)).reshape(x_size)
            else:
                x = layer(x)

        return x


class Conv1DNBM(torch.nn.Module):
    """
    Neural Basis Model using Conv1D outputs as features.

    Args:
        orders (list(tuple(int, int, Optional[int]))): List of orders, corresponding number of bases, and top-k-per-filter
                                                       terms to use for feature interactions in the model.
        sequence_channels (int): Number of feature channels in sequence input.
        sequence_kernels (list): List of sequence kernels, each specified as a tuple of (number of kernels, kernel size)
        padding (str): Padding type for convolution ("same" or "valid")
        pooling (str): Type of pooling to apply after convolution ("max", "avg", or None)
        pooling_size (int): Window size for the pooling operation.

        positional_encoding (str): Type of positional encoding to use ("sinusoidal" or "index")
        positional_encoding_max_len (int): Maximum length of the sequence (if sinusoidal positional encoding).
        positional_encoding_embed_dim (int): Dimension of the positional encoding (if sinusoidal positional encoding).
        positional_encoding_freq_scale (float): Frequency scaling factor (if sinusoidal positional encoding).
        positional_encoding_integral_periods (Optional[List[int]]): List of integral periods to include in the encoding

        conv_activation (str): Activation function to apply after convolution ("relu", "elu", "sigmoid", or None)
        conv_batch_norm (bool): Whether to apply batch normalization after convolution.
        conv_kernel_l1_penalty (float): L1 penalty for the convolutional kernel weights.
        conv_kernel_l2_penalty (float): L2 penalty for the convolutional kernel weights.
        conv_activity_l1_penalty (float): L1 penalty for the convolutional layer activity.
        conv_activity_l2_penalty (float): L2 penalty for the convolutional layer activity.

        feature_hidden_num (int): Number of hidden layers in the feature NN.
        feature_hidden_size (int): Size of the hidden layers in the feature NN.
        feature_activation (str): Activation function for the hidden layers in the feature NN ("relu" or "sigmoid").
        feature_norm (str): Normalization layer to apply after each hidden layer in the feature NN ("batch" or "none").

        tuner_hidden_num: Number of tuner hidden layers,
        tuner_hidden_size: Size of tuner hidden layers,
        tuner_hidden_activation: Activation function between tuner hidden layers,
        tuner_hidden_norm: Normalization layer between tuner hidden layers,

        output_size (int): Number of output features.
        output_activations (List[str]): Activation functions to apply after the output layer for each feature ("relu" or "sigmoid").

        feature_nn_bases_l1_penalty (float): L1 penalty for the per-feature bases weights
        feature_nn_bases_l2_penalty (float): L2 penalty for the per-feature bases weights
        feature_nn_activity_l1_penalty (float): L1 penalty for the feature NN activity.
        feature_nn_activity_l2_penalty (float): L2 penalty for the feature NN activity.
    """

    def __init__(
        self,
        orders: List[Tuple[int, int, Optional[int]]],
        sequence_channels: int,
        padding: str,
        pooling: Optional[str],
        pooling_size: Optional[int],
        sequence_kernels: List[Tuple[int, int]],
        positional_encoding: str,
        positional_encoding_max_len: Optional[int],
        positional_encoding_embed_dim: Optional[int],
        positional_encoding_freq_scale: Optional[float],
        positional_encoding_integral_periods: Optional[List[int]],
        conv_activation: Optional[str],
        conv_batch_norm: bool,
        conv_kernel_l1_penalty: float,
        conv_kernel_l2_penalty: float,
        conv_activity_l1_penalty: float,
        conv_activity_l2_penalty: float,
        feature_hidden_num: int,
        feature_hidden_size: int,
        feature_activation: str,
        feature_norm: str,
        tuner_hidden_num: int,
        tuner_hidden_size: int,
        tuner_hidden_activation: str,
        tuner_hidden_norm: str,
        output_size: int,
        output_activations: List[str],
        feature_nn_bases_l1_penalty: float,
        feature_nn_bases_l2_penalty: float,
        feature_nn_activity_l1_penalty: float,
        feature_nn_activity_l2_penalty: float,
        **kwargs,
    ):
        super(Conv1DNBM, self).__init__()

        self.seq_conv = Conv1D(
            input_channels=sequence_channels,
            kernels=sequence_kernels,
            padding=padding,
            pooling=pooling,
            pooling_size=pooling_size,
            batch_norm=conv_batch_norm,
            activation=conv_activation,
        )

        self.orders = orders
        self.output_size = output_size

        if positional_encoding == "sinusoidal":
            self.positional_encoding = SineCosinePositionalEncoding(
                max_len=positional_encoding_max_len,
                embed_dim=positional_encoding_embed_dim,
                freq_scale=positional_encoding_freq_scale,
            )
            self.feature_size = 1 + positional_encoding_embed_dim
        elif positional_encoding == "hybrid_sine_cosine":
            self.positional_encoding = HybridSineCosinePositionalEncoding(
                max_len=positional_encoding_max_len,
                embed_dim=positional_encoding_embed_dim,
                freq_scale=positional_encoding_freq_scale,
                integral_periods=positional_encoding_integral_periods,
            )
            self.feature_size = 1 + positional_encoding_embed_dim
        elif positional_encoding == "index":
            self.positional_encoding = IndexPositionalEncoding()
            self.feature_size = 2
        else:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding}")

        self.conv_kernel_l1_penalty = conv_kernel_l1_penalty
        self.conv_kernel_l2_penalty = conv_kernel_l2_penalty
        self.conv_activity_l1_penalty = conv_activity_l1_penalty
        self.conv_activity_l2_penalty = conv_activity_l2_penalty

        self.num_kernels = sum(
            [num_kernels for num_kernels, _ in sequence_kernels]
        )

        self.feature_nn = torch.nn.ModuleDict()
        self.output_nn = torch.nn.ModuleDict()
        for order in orders:
            self.feature_nn[str(order[0])] = FeatureBasesNN(
                num_features=self.num_kernels,
                feature_size=self.feature_size,
                order=order[0],
                num_bases=order[1],
                feature_hidden_num=feature_hidden_num,
                feature_hidden_size=feature_hidden_size,
                feature_activation=feature_activation,
                feature_norm=feature_norm,
                per_feature_top_k=order[2],
            )

            self.output_nn[str(order[0])] = torch.nn.Linear(
                in_features=utils.num_combinations_with_replacement(
                    self.num_kernels, order[0]
                ),
                out_features=self.output_size,
            )

        self.feature_nn_bases_l1_penalty = feature_nn_bases_l1_penalty
        self.feature_nn_bases_l2_penalty = feature_nn_bases_l2_penalty
        self.feature_nn_activity_l1_penalty = feature_nn_activity_l1_penalty
        self.feature_nn_activity_l2_penalty = feature_nn_activity_l2_penalty

        self.tuners = torch.nn.ModuleDict()
        for output in range(self.output_size):
            self.tuners[str(output)] = ResidualMLP(
                input_size=1,
                hidden_num=tuner_hidden_num,
                hidden_size=tuner_hidden_size,
                hidden_activation=tuner_hidden_activation,
                hidden_norm=tuner_hidden_norm,
                output_size=1,
                output_activation=output_activations[output],
            )

        self.reg_penalty = dict()

    def get_regularization_penalty(self) -> torch.Tensor:
        """
        Compute the total regularization penalty for the model.

        Returns:
            torch.Tensor: Total regularization penalty.
        """

        return (
            self.conv_kernel_l1_penalty * self.reg_penalty["conv_kernel_l1"]
            + self.conv_kernel_l2_penalty * self.reg_penalty["conv_kernel_l2"]
            + self.conv_activity_l1_penalty * self.reg_penalty["conv_activity_l1"]
            + self.conv_activity_l2_penalty * self.reg_penalty["conv_activity_l2"]
            + self.feature_nn_bases_l1_penalty * self.reg_penalty["feature_nn_bases_l1"]
            + self.feature_nn_bases_l2_penalty * self.reg_penalty["feature_nn_bases_l2"]
            + self.feature_nn_activity_l1_penalty
            * self.reg_penalty["feature_nn_activity_l1"]
            + self.feature_nn_activity_l2_penalty
            * self.reg_penalty["feature_nn_activity_l2"]
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural additive model.
        """
        # Output is (batch_size, num_kernels, input_size)

        x_1 = self.seq_conv(seq)

        batch_size = x_1.size(0)
        input_size = x_1.size(2)

        x_2 = self.positional_encoding(
            x_1.reshape(batch_size, self.num_kernels, 1, input_size)
        )

        x_3_list = []
        x_4_list = []
        for order in self.orders:
            num_feat_combs = utils.num_combinations_with_replacement(
                self.num_kernels,
                order[0],
            )
            x_3_feat = self.feature_nn[str(order[0])](x_2)

            num_pos_combs = x_3_feat.shape[2]

            x_3_list.append(x_3_feat)

            x_4_feat = self.output_nn[str(order[0])](
                x_3_feat.permute(0, 2, 1).reshape(
                    batch_size * num_pos_combs, num_feat_combs
                )
            ).reshape(batch_size, num_pos_combs, self.output_size)
            x_4_list.append(x_4_feat)

        x_4 = torch.cat(x_4_list, dim=1).sum(dim=1)  # (batch_size, output_size)

        x_5 = torch.zeros(
            batch_size,
            self.output_size,
            device=x_4.device,
        )

        # Apply the tuners to the output
        for i in range(self.output_size):
            x_5[:, i] = (
                self.tuners[str(i)](
                    x_4[:, i].unsqueeze(1).reshape(batch_size, 1)
                ).squeeze(1)
                + x_4[:, i]
            ) # Residual connection

        # Compute regularization losses only during training
        if not self.training:
            self.reg_penalty = dict()
        else:
            # Compute regularization losses for the convolutional layer
            seq_conv_kernel_l1_loss, seq_conv_kernel_l2_loss = (
                self._compute_weight_penalties(self.seq_conv.convs, x_1.device)
            )
            
            conv_kernel_l1_loss = seq_conv_kernel_l1_loss
            conv_kernel_l2_loss = seq_conv_kernel_l2_loss

            conv_activity_l1_loss = torch.norm(x_1, p=1)
            conv_activity_l2_loss = torch.norm(x_1, p=2)

            # Compute regularization losses for the feature NBMs
            feature_nn_bases_l1_loss, feature_nn_bases_l2_loss = tuple(
                map(
                    sum,
                    zip(
                        *[
                            self._compute_weight_penalties(
                                feat_nn.shape_fn_nn, x_1.device
                            )
                            for ord, feat_nn in self.feature_nn.items()
                        ]
                    ),
                )
            )

            feature_nn_activity_l1_loss = sum(
                [torch.norm(x_3_feat, p=1) for x_3_feat in x_3_list]
            )
            feature_nn_activity_l2_loss = sum(
                [torch.norm(x_3_feat, p=2) for x_3_feat in x_3_list]
            )

            # Do not return directly, but store these values to be retrieved by the training loop later
            self.reg_penalty = {
                "conv_kernel_l1": conv_kernel_l1_loss,
                "conv_kernel_l2": conv_kernel_l2_loss,
                "conv_activity_l1": conv_activity_l1_loss,
                "conv_activity_l2": conv_activity_l2_loss,
                "feature_nn_bases_l1": feature_nn_bases_l1_loss,
                "feature_nn_bases_l2": feature_nn_bases_l2_loss,
                "feature_nn_activity_l1": feature_nn_activity_l1_loss,
                "feature_nn_activity_l2": feature_nn_activity_l2_loss,
            }

        return x_5

    def _compute_weight_penalties(
        self, layers: torch.nn.Sequential, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute L1 and L2 penalties for the weights of the given layers.

        Args:
            layers (torch.nn.Sequential): The layers to compute penalties for.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The L1 and L2 penalties.
        """

        l1_loss = torch.tensor(0.0, device=device)
        l2_loss = torch.tensor(0.0, device=device)
        for layer in layers:
            if isinstance(layer, torch.nn.Linear):
                l1_loss += torch.norm(layer.weight, p=1)
                l2_loss += torch.norm(layer.weight, p=2)
            if isinstance(layer, torch.nn.Conv1d):
                l1_loss += torch.norm(layer.weight, p=1)
                l2_loss += torch.norm(layer.weight, p=2)
        return l1_loss, l2_loss
