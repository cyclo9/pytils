import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

    def forward(self, queries, keys, values):
        """
        `queries`: what you're considering; questions
        `keys`: what to compare to `queries` to see which are relevant
        `values`: info to be aggregated or used; could be the same as `keys`

        `keys` and `values` can have longer sequence lengths than `queries`
        """
        return self.attn(queries, keys, values)[0]
