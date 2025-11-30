import torch


class IndexPositionalEncoding(torch.nn.Module):
    """
    Positional encoding for sequences using index-based encoding.
    """

    def __init__(self):
        """
        Initialize the positional encoding.
        """
        super(IndexPositionalEncoding, self).__init__()

    def get_position_encoding(self, pos: int):
        # Get the positional encoding vector at position pos
        return torch.tensor([pos])

    def forward(self, x):
        assert len(x.shape) == 4

        # Create a positional encoding based on the index of the elements
        batch_size, num_features, _, seq_length = x.size()
        positions = (
            torch.arange(seq_length, device=x.device)
            .view(1, 1, 1, -1)
            .expand(batch_size, num_features, -1, -1)
            .clone()
        )

        # Concatenate the positional encoding to x in the 2nd dimension
        x_pe = torch.cat((x, positions), dim=2)

        return x_pe
