from collections import defaultdict
import copy
import json
from itertools import batched, product, islice
import logomaker
import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import random
import scipy
from scipy import stats
import sys
import torch
from typing import List, Tuple, Iterator, Generator, Optional
from tqdm.auto import tqdm
import warnings

# Local imports
from datasets import sequence
from experiment import experiment
from metrics import metrics, losses
from utils import utils


def _sample_kmers_generator(sequences: List[str], k: int, frac: float, seed: int):
    """
    Generate an iterator that samples k-mers seen in a list of sequences.

    Args:
        sequences (List[str]): List of sequences from which to collect kmers.
        k (int): Size of k-mer
        frac (float): Fraction of k-mers to sample for the generator
        seed (int): Seed for sampling
    """
    kmers = {seq[idx : idx + k] for seq in sequences for idx in range(len(seq) - k + 1)}
    random.seed(seed)
    sampled_kmers = random.sample(list(kmers), k=int(frac * len(kmers)))
    yield from sampled_kmers


def _all_kmers_generator(alphabet: List[str], k: int):
    """
    Generate all possible kmers of length k given an alphabet.

    Args:
        alphabet (List[str]): Alphabet of bases
        k (int): Length of kmer
    """
    choices = [alphabet for i in range(k)]
    for x in product(*choices):
        yield "".join(x)


def _random_kmers_generator(alphabet: List[str], k: int, n: int, seed: int) -> List[str]:
    """
    Generate a list of n random kmers of length k given an alphabet.

    Args:
        alphabet (List[str]): Alphabet of bases.
        k (int): Length of each kmer.
        n (int): Number of random kmers to generate.

    Returns:
        List[str]: List of n random kmers.
    """
    random.seed(seed)
    return [''.join(random.choices(alphabet, k=k)) for _ in range(n)]


def _one_hot_encode_torch(seqs: List[str], alphabet: List[str]) -> torch.Tensor:
    """
    Vectorized one-hot encoding of a batch of sequences into a PyTorch tensor.
    
    Args:
        seqs (List[str]): List of equal-length sequences (e.g., kmers)
        alphabet (List[str]): Alphabet used for encoding, e.g. ["A", "C", "G", "T"]
    
    Returns:
        Tensor of shape (batch_size, seq_len, alphabet_size)
    """
    char_to_int = {char: i for i, char in enumerate(alphabet)}
    seq_len = len(seqs[0])
    batch_size = len(seqs)
    vocab_size = len(alphabet)

    # Convert sequences to integer indices
    int_encoded = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i, seq in enumerate(seqs):
        int_encoded[i] = [char_to_int[char] for char in seq]

    # One-hot encode
    one_hot = np.eye(vocab_size)[int_encoded]  # shape: (batch_size, seq_len, vocab_size)
    return torch.from_numpy(one_hot).float()


def _compute_weighted_freqs(
    kmers: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute the weighted position-specific frequency matrix of kmers.
    Returns a matrix of shape (positions, alphabet size).

    Args:
        kmers (np.ndarray): Array of one-hot-encoded kmers (dim: [num_kmers, kmer_length, channels])
        weights (np.ndarray): Weights for each kmer
    """
    kmer_one_hot = kmers.transpose(0, 2, 1)  # (num_kmers, channels, kmer_length)
    kmer_weights = (weights.reshape(-1, 1, 1) * kmer_one_hot) ** 2
    total_weights = np.sum(kmer_weights, axis=0)
    return total_weights / np.sum(
        total_weights, axis=0, keepdims=True
    )  # Freq matrix (positions, alphabet size)


def _compute_sequence_logo_heights(
    freqs: np.ndarray, alphabet: List[str], n: int, eps: float = 1e-6
) -> pd.DataFrame:
    """
    Calculates the height of each base in the consensus logo:

    The height of each base in a position is given by
    height = f_{b_i} * R_i

    where f_{b_i} is the relative frequency of base b at position i
    and R_i is the information content of position i, given by

    R_i = log_2(4) - (H_i + e_n)

    where H_i is the uncertainty / Shannon entropy of position i, given by

    H_i = -\\Sigma_{b=1}^{t} f_{b,i} * log_2(f_{b_i})

    and e_n is the small-sample corection, where n is the number of kmers,

    e_n = \\frac{1}{ln 2} * \frac{4 - 1}{2n}

    Args:
        freqs (np.ndarray): Position-specific frequency matrix of kmers. Dim: [num_bases, sequence_length]
        alphabet (List[str]): Ordered alphabet to map sequence characters to indices
        n (int): Number of kmers
        eps (float): Small value to prevent log(0)
    """

    H = (freqs * -np.log2(freqs + eps)).sum(
        axis=0
    )  # Yields a vector of Shannon entropies. Dim: [sequence_length]
    e = 1 / np.log(2) * (len(alphabet) - 1) / (2 * n)  # Small-sample correction
    R = np.log2(len(alphabet)) - (
        H + e
    )  # Yields a vector of information content. Dim: [sequence_length]

    heights = freqs * R  # Yields a matrix of heights. Dim: [sequence_length, num_bases]

    return pd.DataFrame(heights.T, columns=alphabet)


def compute_conv1d_activations_batched(
    conv_layer: torch.nn.modules.conv.Conv1d,
    activation_layer: Optional[torch.nn.Module],
    batchnorm_layer: Optional[torch.nn.BatchNorm1d],
    alphabet: List[str],
    kmer_generator: Generator[str, None, None],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes activations for all possible kmers given a convolutional layer.
    Returns a numpy array of one-hot-encoded kmers (dim: [num_kmers, kmer_length,
    alphabet_size]) and a numpy array of per-kernel kmer activation values (dim:
    [num_kmers, num_kernels, 1]).

    Args:
        conv_layer (torch.nn.modules.conv.Conv1d): Convolutional layer to compute activations for
        activation_layer (torch.nn.Module): Activation layer to apply to the outputs of the convolutional layer
        alphabet (List[str]): Ordered alphabet to map sequence characters to indices
        kmer_generator (List[str]): Iterator for all kmers to use in computing activations
        device (torch.device): Device to use for computation
        batch_size (int): Batch size to use for computation
    """
    # Ensure the kernel is 1D
    assert len(conv_layer.kernel_size) == 1

    all_encoded = []
    all_activations = []
    with torch.no_grad():
        for kmer_batch in tqdm(batched(kmer_generator, batch_size)):
            encoded_batch = _one_hot_encode_torch(kmer_batch, alphabet)
            x1 = encoded_batch.permute(0, 2, 1).to(device)
            x2 = torch.nn.functional.conv1d(
                input=x1,
                weight=conv_layer.weight,
                bias=conv_layer.bias,
                stride=conv_layer.stride,
                padding=0,
                dilation=conv_layer.dilation,
                groups=conv_layer.groups,
            )

            if batchnorm_layer is not None:
                x2 = batchnorm_layer(x2)

            if activation_layer is not None:
                x2 = activation_layer(x2)

            all_encoded.append(encoded_batch)
            all_activations.append(x2.cpu())

    encoded_array = torch.cat(all_encoded, dim=0).numpy()
    activation_array = torch.cat(all_activations, dim=0).numpy()
            
    return encoded_array, activation_array


def compute_conv1d_activations(
    conv_layer: torch.nn.modules.conv.Conv1d,
    activation_layer: torch.nn.Module,
    batchnorm_layer: torch.nn.BatchNorm1d,
    alphabet: List[str],
    kmer_generator: Generator[str, None, None],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes activations for all possible kmers given a convolutional layer.
    Returns a numpy array of one-hot-encoded kmers (dim: [num_kmers, kmer_length,
    alphabet_size]) and a numpy array of per-kernel kmer activation values (dim:
    [num_kmers, num_kernels, 1]).

    Args:
        conv_layer (torch.nn.modules.conv.Conv1d): Convolutional layer to compute activations for
        activation_layer (torch.nn.Module): Activation layer to apply to the outputs of the convolutional layer
        alphabet (List[str]): Ordered alphabet to map sequence characters to indices
        kmer_generator (List[str]): Iterator for all kmers to use in computing activations
        device (torch.device): Device to use for computation
    """
    # Ensure the kernel is 1D
    assert len(conv_layer.kernel_size) == 1

    encode_fn = _one_hot_encode_vector_fn(alphabet)
    encoded_seqs = torch.stack([encode_fn(seq) for seq in kmer_generator], dim=0)

    with torch.no_grad():
        x1 = encoded_seqs.permute(0, 2, 1).to(device)
        x2 = torch.nn.functional.conv1d(
            input=x1,
            weight=conv_layer.weight,
            bias=conv_layer.bias,
            stride=conv_layer.stride,
            padding=0,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
        )
        x3 = batchnorm_layer(x2)
        x4 = activation_layer(x3)

    return encoded_seqs.cpu().numpy(), x4.cpu().numpy()


def plot_sequence_logo(
    kernel_activations: np.ndarray,
    kmers: np.ndarray,
    alphabet: List[str],
    alphabet_colors: dict[str, str],
    ax: matplotlib.axes.Axes,
    min_activation: float = 0,
    max_activation: float = np.inf,
    **kwargs,
) -> None:
    """
    Plots a sequence logo of kmer activations.

    Args:
        kernel_activations (np.array): Array of kernel activation values
        kmers (np.array): Array of one-hot-encoded kmers corresponding to activations
        alphabet (List[str]): Ordered alphabet to map sequence characters to indices
        alphabet_colors (dict[str, str]): Dictionary mapping bases to colors
        ax (matplotlib.axes.Axes): Axes to plot the sequence logo on
        min_activation (float): Minimum activation threshold for kmer inclusion
        **kwargs: Additional arguments to pass to logomaker.Logo
    """

    if not np.any(kernel_activations):
        warnings.warn("All k-mer activations were zero. Sequence logo will be empty.")
        return

    seq_idxs = (kernel_activations >= min_activation) & (kernel_activations <= max_activation)
    
    freqs = _compute_weighted_freqs(
        kmers=kmers[seq_idxs],
        weights=kernel_activations[seq_idxs],
    )

    heights = _compute_sequence_logo_heights(
        freqs=freqs,
        alphabet=alphabet,
        n=len(kmers),
    )

    logomaker.Logo(
        df=heights,
        color_scheme=alphabet_colors,
        ax=ax,
        **kwargs,
    )
    

def get_model_predictions(params, model, dataset, device, 
                          activation_fn, frac=1):
    """
    Get predictions from the model for a given dataset.

    Args:
        params (dict): Experiment parameters.
        model (torch.nn.Module): Trained model.
        dataset (pd.DataFrame): Input dataset.
        device (torch.device): Device to run computations.
        frac (float): Fraction of the dataset to sample.

    Returns:
        Tuple: labels, predictions, and prediction variances.
    """
    seq_data = sequence.SequenceDataset(
        data_path=None,
        data_df=dataset.sample(frac=frac),
        sequence_column=params["sequence_column"],
        target_column=params["target_column"],
        upstream_sequence=params["upstream_sequence"],
        downstream_sequence=params["downstream_sequence"],
    )
    loader = torch.utils.data.DataLoader(seq_data, batch_size=2048, shuffle=False)

    labels, preds, pred_vars = [], [], []
    with torch.no_grad():
        for seq, y in tqdm(loader):
            seq = seq.to(device)
            out = model(seq).cpu().numpy()
            labels.append(y.cpu().numpy())
            preds.append(out[:, 0])
            #pred_vars.append(out[:, 1])

    return (
        np.concatenate(labels),
        activation_fn(np.concatenate(preds).flatten()),
        #np.concatenate(pred_vars).flatten(),
    )

def plot_hex_histogram(fig, ax, x, y, gridsize=30, linewidths=0):
    # Hexbin plot
    hb = ax.hexbin(
        x,
        y,
        bins="log",
        cmap="viridis",
        mincnt=1,
        gridsize=gridsize,
        edgecolors='none',
        linewidths=linewidths
    )
    
    r = stats.linregress(x, y)
    ax.text(0.025, 0.975, f"n = {len(x)}\nr = {r.rvalue:.3f}", ha="left", va="top", transform=ax.transAxes)
    fig.colorbar(hb, ax=ax, label="")
    
    return fig, ax
    

def calculate_shape_logits(
    model: torch.nn.Module,
    feat_pos_activations: torch.Tensor,  # [batch x num terms x 3] dim tensor
    output_idx: int,
) -> torch.Tensor:
    """
    Calculate logits from learned shape function corresponding to the
    f(activation_{feat_i, pos_i} for feat_i, pos_i in feat_pos_activations)

    Arguments:
    - model (torch.nn.Module): Trained NBM model
    - feat_pos_activations (torch.tensor): [batch x num_terms x 2 + (model.feature_size)] dim
      where num_terms is the order of the shape function. For each term:
        - First element: feature index.
        - Second element: actual input score.
        - [Third:last] element(s): positional encoding.
    """
    # Validate input tensor dimensions
    assert len(feat_pos_activations.shape) == 3
    assert feat_pos_activations.shape[2] == 1 + model.feature_size

    batch_size = feat_pos_activations.shape[0]
    order = feat_pos_activations.shape[1]

    # Ensure feature and position indices are identical across batches
    for i in range(order):
        assert torch.all(feat_pos_activations[:, i, 0] == feat_pos_activations[0, i, 0])
        # Positional encoding can be multi-dimensional
        assert torch.all(
            feat_pos_activations[:, i, 2:] == feat_pos_activations[0, i, 2:]
        )

    # Extract feature indices
    feat_idxs = feat_pos_activations[0, :, 0].int().tolist()

    # Ensure feature indices are sorted for correct indexing
    assert torch.equal(
        feat_pos_activations[0, :, 0], torch.sort(feat_pos_activations[0, :, 0])[0]
    )

    # Retrieve model parameters
    num_kernels = 0
    if hasattr(model, "seq_conv"):
        num_kernels += sum(
            [conv.out_channels for conv in model.seq_conv.convs.values()]
        )

    # Form the input tensor to the feature_nn
    feature_nn = model.feature_nn[str(order)]
    x_in = feat_pos_activations[:, :, 1:].reshape(-1, order * model.feature_size)

    # Apply the base network
    x_out_base = feature_nn.base_nn(x_in)

    # Apply the feature network
    num_combos = utils.num_combinations_with_replacement(num_kernels, order)
    feat_rank = utils.rank_combination_with_replacement(num_kernels, order, feat_idxs)

    # Prepare input for the feature network
    x_in_feat = torch.zeros(
        (
            batch_size,
            1,
            feature_nn.num_bases * num_combos,
        ),
        device=feat_pos_activations.device,
    )

    x_in_feat[
        :, 0, feature_nn.num_bases * feat_rank : feature_nn.num_bases * (feat_rank + 1)
    ] = x_out_base
    x_in_feat = x_in_feat.permute(0, 2, 1)

    # Apply the feature network
    x_out_feat = feature_nn.shape_fn_nn(x_in_feat)[:, feat_rank].flatten()

    # Prepare input for the output layer
    x_in_out = torch.zeros((batch_size, num_combos), device=feat_pos_activations.device)
    x_in_out[:, feat_rank] = x_out_feat

    # Apply the output layer
    x_out_out = model.output_nn[str(order)](x_in_out)[:, output_idx]

    return x_out_out


def calculate_shape_logits_batched(
    model: torch.nn.Module,
    feat_pos_activations: torch.Tensor,
    output_idx: int,
    minibatch_size: int = 8192,
) -> torch.Tensor:
    """
    Splits feat_pos_activations into smaller batches, runs each through the model,
    and concatenates the outputs.

    Args:
        model (torch.nn.Module): The PyTorch model to apply.
        feat_pos_activations (torch.Tensor): The input tensor of shape [batch_size, ...].
        minibatch_size (int): The size of the smaller batches.

    Returns:
        torch.Tensor: The concatenated model outputs of shape [batch_size, ...].
    """
    outputs = []
    for i in range(0, feat_pos_activations.size(0), minibatch_size):
        batch = feat_pos_activations[i : i + minibatch_size]
        with torch.no_grad():
            out = calculate_shape_logits(model, batch, output_idx)
        outputs.append(out)

    return torch.cat(outputs, dim=0)


def calculate_first_order_shape_functions(
    seq_activations: torch.Tensor,
    model: torch.nn.Module,
    output_idx: int,
    num_positions: int,
    device: torch.device,
    linspace_num: int = 100,
    kmer_sample_num: int = 1000,
    activation_fill: float = -1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate shape functions for sequence activations.

    Args:
        seq_activations (torch.Tensor): Input activations of shape [num_kmers, num_seq_filters, 1].
        model (torch.nn.Module): Trained model to calculate logits.
        num_positions (int): Number of positions to evaluate.
        device (torch.device): Device to perform computations on.
        linspace_num (int): Number of points for linear interpolation.
        kmer_sample_num (int): Number of points to sample for kmer activations
        activation_fill (float): Filler value for activations

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for kmer contributions and linspace contributions.
    """

    num_kmers = seq_activations.shape[0]
    num_seq_filters = seq_activations.shape[1]

    actual_seq_filter_idxs = []
    actual_positions = []
    actual_features = []
    actual_contributions = []

    linspace_seq_filter_idxs = []
    linspace_positions = []
    linspace_features = []
    linspace_contributions = []

    for seq_filter_idx in range(num_seq_filters):
        # Extract activations for the current filter
        filter_activations = seq_activations[:, seq_filter_idx, 0]
        activations_linspace = torch.linspace(
            torch.min(filter_activations), torch.max(filter_activations), linspace_num
        )

        for position in range(num_positions):
            # Calculate shape function in the actual domain
            kmer_activations = torch.full(
                (kmer_sample_num, 1, 1 + model.feature_size),
                activation_fill,
                device=device,
            )
            kmer_activations[:, 0, 0] = seq_filter_idx
            # Use a random sample of the kmers
            random_indices = torch.randint(
                0, filter_activations.shape[0], (kmer_sample_num,)
            )
            kmer_activations[:, 0, 1] = filter_activations[random_indices]
            kmer_activations[:, 0, 2:] = (
                model.positional_encoding.get_position_encoding(position)
            )

            kmer_logits = calculate_shape_logits_batched(
                model, kmer_activations, output_idx
            )
            mean_kmer_logit = torch.mean(kmer_logits)
            adj_kmer_logits = kmer_logits - mean_kmer_logit

            actual_seq_filter_idxs.append(kmer_activations[:, 0, 0].cpu())
            actual_features.append(kmer_activations[:, 0, 1].cpu())
            actual_positions.append(torch.full((kmer_sample_num,), position))
            actual_contributions.append(adj_kmer_logits.cpu())

            # Calculate shape function in the linear interpolation domain
            linspace_activations = torch.full(
                (linspace_num, 1, 1 + model.feature_size),
                activation_fill,
                device=device,
            )
            linspace_activations[:, 0, 0] = seq_filter_idx
            linspace_activations[:, 0, 1] = activations_linspace
            linspace_activations[:, 0, 2:] = (
                model.positional_encoding.get_position_encoding(position)
            )

            linspace_logits = calculate_shape_logits_batched(
                model, linspace_activations, output_idx
            )
            # Also use the mean of the kmer activations
            adj_linspace_logits = linspace_logits - mean_kmer_logit

            linspace_seq_filter_idxs.append(linspace_activations[:, 0, 0].cpu())
            linspace_features.append(linspace_activations[:, 0, 1].cpu())
            linspace_positions.append(torch.full((linspace_num,), position))
            linspace_contributions.append(adj_linspace_logits.cpu())

    # Create DataFrame for kmer contributions
    kmer_contribution_df = pd.DataFrame(
        {
            "seq_filter_idx": torch.stack(actual_seq_filter_idxs).flatten(),
            "position": torch.stack(actual_positions).flatten(),
            "feature": torch.stack(actual_features).flatten(),
            "adj_logit": torch.stack(actual_contributions).flatten(),
        }
    )

    linspace_contribution_df = pd.DataFrame(
        {
            "seq_filter_idx": torch.stack(linspace_seq_filter_idxs).flatten(),
            "position": torch.stack(linspace_positions).flatten(),
            "feature": torch.stack(linspace_features).flatten(),
            "adj_logit": torch.stack(linspace_contributions).flatten(),
        }
    )

    return kmer_contribution_df, linspace_contribution_df


def calculate_feature_nam_ablations(
    params, model, dataset, num_features, metric_fns, device, activation_fn
):
    def ablation_pre_hook(module, input, feature_idx: int = 0):
        """
        Replaces the values in the `feature_idx` column of the input tensor
        with the mean of that column (i.e., ablation).
        """
        x = input[0]  # input is a tuple of tensors
        x = x.clone()  # avoid modifying input in-place
    
        # Compute the mean of the feature to ablate
        feat_mean = x[:, feature_idx].mean()
    
        # Replace the column with the mean
        x[:, feature_idx] = feat_mean
    
        # Return a new tuple with the modified tensor
        return (x,)

    ablation_performances = dict()
    for feature_idx in range(num_features):
        ablation_hook = model.output_nn["1"].register_forward_pre_hook(
            lambda module, input: ablation_pre_hook(
                module, input, feature_idx
            )
        )

        # Run predictions over entire dataset
        with torch.no_grad():
            labels, preds = get_model_predictions(params, model, dataset, device, activation_fn)

        # Track metrics
        labels = torch.tensor(labels)
        preds = torch.tensor(preds) 
        
        ablation_performances[feature_idx] = {
            metric: fn(preds, labels) for metric, fn in metric_fns.items()
        }
        
        ablation_hook.remove()

    return ablation_performances


def plot_kernel(
    conv_weight,
    kmers,
    seq_activations,
    linspace_contribution_df,
    num_seq_filters,
    num_positions,
    seq_filter_idx,
    logo_q=(0.0, 1.0),
    conv_kernel_ordering=None,
    title="",
    label="",
    trim_left=0,   # number of positions to trim from the left
    trim_right=0,  # number of positions to trim from the right
    tuner_left=0,
    tuner_right=0,
    figsize=(5,4),
    fontsize=10,
    display_y=True
):
    """Plot a sequence logo and tuner contribution (top-bottom difference) for a given kernel,
    with optional manual trimming of the left and right positions.
    """

    conv_kernel_labels = {}

    # Map kernel index through optional ordering
    seq_filter_idx = (
        conv_kernel_ordering[seq_filter_idx]
        if conv_kernel_ordering is not None
        else seq_filter_idx
    )

    # Convert convolution weights to numpy
    conv_weight_np = conv_weight.detach().cpu().numpy()

    # Group linspace contributions by sequence filter
    grouped_linspace = linspace_contribution_df.groupby("seq_filter_idx")

    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=600)

    # --- Sequence logo subplot ---
    filter_activations = seq_activations[:, seq_filter_idx].flatten()
    min_act = np.quantile(filter_activations, logo_q[0])
    max_act = np.quantile(filter_activations, logo_q[1])

    plot_sequence_logo(
        kernel_activations=filter_activations,
        kmers=kmers,
        alphabet=["A", "C", "G", "T"],
        alphabet_colors="classic",
        ax=ax[0],
        min_activation=min_act,
        max_activation=max_act,
    )

    ax[0].set_ylim(0, 2)
    ax[0].set_yticks([])
    ax[0].set_xlim(trim_left, kmers.shape[1] - trim_right)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)

    # --- Contribution difference subplot ---
    if seq_filter_idx not in grouped_linspace.groups:
        ax[1].text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    linspace_df = grouped_linspace.get_group(seq_filter_idx)
    grid_df = (
        linspace_df.pivot(index="feature", columns="position", values="adj_logit")
        .sort_index()
        .sort_index(axis=1)
    )

    grid = grid_df.to_numpy()
    x_vals = grid_df.columns.to_numpy()
    diff = grid[-1, :] - grid[0, :] if grid.shape[0] >= 2 else grid[0, :]

    # Trim contribution difference plot 
    start = tuner_left
    end = len(diff) - tuner_right if tuner_right > 0 else len(diff)
    diff_window = diff[start:end]
    x_window = np.arange(len(diff_window))

    pos_mask = diff_window > 0
    neg_mask = diff_window < 0

    ax[1].fill_between(
        x_window, 0, diff_window, where=pos_mask, interpolate=True, color="tab:blue", alpha=0.5, linewidth=0
    )
    ax[1].fill_between(
        x_window, 0, diff_window, where=neg_mask, interpolate=True, color="tab:red", alpha=0.5, linewidth=0
    )
    ax[1].plot(x_window, diff_window, color="black", linewidth=0.6)
    ax[1].set_ylim(-12, 12)
    ax[1].set_xlim(0, len(x_window))
    #ax[1].set_ylabel("Score", fontsize=fontsize)
    #ax[1].set_xlabel("Sequence Position", fontsize=fontsize)
    ax[0].text(0.975, 0.95, label, ha="right", va="top", transform=ax[0].transAxes, fontsize=fontsize-2)
    ax[0].set_title(title, fontsize=fontsize)
    ax[1].set_xticks([0, len(x_window)])
    ax[1].spines[["top", "right"]].set_visible(False)

    ax[1].tick_params(axis='x', labelsize=fontsize)
    ax[1].tick_params(axis='y', labelsize=fontsize)

    if not display_y:
        ax[1].set_yticklabels([])

    return fig


def get_tuner_domain(model, df, params, device, output_idx):
    model_copy = copy.deepcopy(model)
    model_copy.tuners[str(output_idx)] = torch.nn.Identity()

    dataset = sequence.SequenceDataset(
        data_path=None,
        data_df=df,
        sequence_column=params["sequence_column"],
        target_column=params["target_column"],
        upstream_sequence=params["upstream_sequence"],
        downstream_sequence=params["downstream_sequence"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    out_min = torch.inf
    out_max = -torch.inf
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader):
            seq, _ = batch
            seq = seq.to(device)
            out = model_copy(seq).cpu().numpy()[:, output_idx]
            all_outputs.append(out)
            out_min = min(out_min, np.min(out))
            out_max = max(out_max, np.max(out))

    all_outputs = np.concatenate(all_outputs, axis=0).squeeze()
    return (out_min, out_max), all_outputs


def plot_tuner(
    model, title, domain=(-5, 5), num_points=1000, output_samples=None, device="cpu", sigmoid=True
):
    model.to(device)
    model.eval()

    # Evaluate model over a grid
    x_np = np.linspace(domain[0], domain[1], num_points, dtype=np.float32)
    x = torch.from_numpy(x_np).unsqueeze(1).to(device)

    with torch.no_grad():
        # Tuner has a residual connection
        if sigmoid:
            y = torch.sigmoid(x + model(x)).squeeze().cpu().numpy()
        else:
            y = (x + model(x)).squeeze().cpu().numpy()

    print("Model input/output shapes:", x.shape, y.shape)

    # Create figure with 2 subplots: function plot + histogram
    fig, ax = plt.subplots(
        1, 1, figsize=(3.0, 2.1), dpi=600,
    )

    # Plot model output
    ax.plot(x_np, y)
    ax.set_xlabel("Total Feature Score")
    ax.set_ylabel("Final Prediction")
    ax.set_title(title)
    ax.set_xlim(-50, 50)
    ax.grid(True)

    #fig.tight_layout()
    return fig
