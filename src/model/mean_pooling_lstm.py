import torch.nn as nn


class MeanPoolingLSTM(nn.Module):
    """
    LSTM-based temporal feature extractor.

    Processes sequence data and uses the mean of the LSTM outputs across all
    time steps as the final sequence representation. Commonly used to encode
    pedestrian or vehicle trajectories.
    """
    def __init__(self, input_dim, embed_dim, layer_num):
        """
        Args:
            input_dim (int): LSTM input feature dimension.
            embed_dim (int): LSTM hidden size, also the output dimension.
            layer_num (int): Number of LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embed_dim,
            num_layers=layer_num,
            batch_first=True,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence tensor.
                Shape: (batch_size, num_agents, seq_len, input_dim)
                or any leading dimensions as long as the last two are
                `(seq_len, input_dim)`.
        
        Returns:
            torch.Tensor: Pooled feature tensor.
                Shape: (batch_size, num_agents, embed_dim), preserving all
                dimensions except the second-to-last `seq_len` dimension.
        """
        shape = x.shape
        x = x.view(-1, *shape[-2:])  # (batch_size * N, seq_len, input_dim)
        out, _ = self.lstm(x)  # (batch_size, seq_len, embed_dim)
        out = out.mean(dim=-2)  # (batch_size, embed_dim)
        out = out.view(*shape[:-2], shape[-1])  # (batch_size, N, embed_dim)
        return out
