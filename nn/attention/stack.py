import torch.nn as nn


class AttentionStack(nn.Module):
    def __init__(self, d_model, n_heads=1, n_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttentionLayer(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, q, k, v, attn_mask=None):
        for block in self.blocks:
            q = block(q, k, v, attn_mask=attn_mask)
        return q


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = MHA(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask=None):
        out = self.mha(q, k, v, attn_mask=attn_mask)
        return self.norm(q + out)


class MHA(nn.Module):
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

    def forward(self, queries, keys, values, attn_mask=None):
        """
        `queries`: what you're considering; questions
        `keys`: what to compare to `queries` to see which are relevant
        `values`: info to be aggregated or used; could be the same as `keys`

        `keys` and `values` can have longer sequence lengths than `queries`
        """
        return self.attn(queries, keys, values, attn_mask=attn_mask)[0]
