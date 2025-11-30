import math
import torch
from typing import List, Optional


class SineCosinePositionalEncoding(torch.nn.Module):
    """
    Positional encoding for sequences using sine and cosine functions.
    """

    def __init__(
        self,
        max_len: int,
        embed_dim: int,
        freq_scale: float = 10000.0,
    ) -> None:
        """
        Initialize the positional encoding.

        Args:
            max_len (int): Length of the sequence.
            embed_dim (int): Dimension of the embedding.
            freq_scale (float): Frequency scaling factor.
        """
        super(SineCosinePositionalEncoding, self).__init__()
        # Compute the positional encodings once
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(freq_scale) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0).unsqueeze(0).unsqueeze(0)

        self.register_buffer("pe", pe)

    def get_position_encoding(self, pos: int):
        # Get the positional encoding vector at position pos
        return self.pe[:, :, :, pos]

    def forward(self, x):
        assert len(x.shape) == 4

        # Concatenate the positional encoding to x in the 2nd dimension
        x_pe = self.pe[:, :, :, : x.size(3)].repeat(x.size(0), x.size(1), 1, 1)

        x = torch.cat((x, x_pe), dim=2)

        return x


class HybridSineCosinePositionalEncoding(torch.nn.Module):
    """
    Positional encoding for sequences using sine and cosine functions, optionally including integral periods.
    """

    def __init__(
        self,
        max_len: int,
        embed_dim: int,
        freq_scale: float = 10000.0,
        integral_periods: Optional[List[int]] = None,
    ):
        """
        Initialize the positional encoding.

        Args:
            max_len (int): Length of the sequence.
            embed_dim (int): Dimension of the embedding.
            freq_scale (int): Frequency scaling factor.
            integral_periods (Optional[List[int]]): List of integral periods to include in the encoding.
        """
        super(HybridSineCosinePositionalEncoding, self).__init__()

        # Validate that there are enough dimensions for the integral periods
        base_dim = embed_dim - 2 * len(integral_periods)
        assert (
            base_dim >= 0 and base_dim % 2 == 0
        ), "embed_dim too small for given integral periods"

        # Compute the positional encodings once
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        # --- Transformer-style encoding (log-spaced frequencies) ---
        if base_dim > 0:
            div_term = torch.exp(
                torch.arange(0, base_dim, 2) * -(math.log(freq_scale) / embed_dim)
            )
            pe[:, 0:base_dim:2] = torch.sin(position * div_term)
            pe[:, 1:base_dim:2] = torch.cos(position * div_term)

        # --- Add integral period frequencies ---
        for i, T in enumerate(integral_periods):
            omega = 2 * math.pi / T
            idx = base_dim + 2 * i
            pe[:, idx] = torch.sin(position.squeeze() * omega)
            pe[:, idx + 1] = torch.cos(position.squeeze() * omega)

        pe = (
            pe.permute(1, 0).unsqueeze(0).unsqueeze(0)
        )  # shape: [1, 1, embed_dim, max_len]
        self.register_buffer("pe", pe)

    def get_position_encoding(self, pos: int):
        # Get the positional encoding vector at position pos
        return self.pe[:, :, :, pos]

    def forward(self, x):
        assert len(x.shape) == 4

        # Concatenate the positional encoding to x in the 2nd dimension
        x_pe = self.pe[:, :, :, : x.size(3)].repeat(x.size(0), x.size(1), 1, 1)

        x = torch.cat((x, x_pe), dim=2)

        return x
