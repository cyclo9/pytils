from common_imports import torch, nn
import math


class AttentionStack(nn.Module):
    def __init__(self, in_size, d_model, num_heads, layers):
        super().__init__()
        blocks = [AttentionBlock(in_size, d_model, num_heads)]
        for _ in range(layers - 1):
            blocks.append(AttentionBlock(d_model, d_model, num_heads))
        self.model = nn.Sequential(*blocks)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        x = self.model(x)
        return self.out(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_size, d_model, num_heads):
        """out Shape: (batch_size, seq_len, d_model)"""

        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.score_norm = math.sqrt(self.d_head)

        # linear layers for q/k/v
        self.W_q = nn.Linear(in_size, d_model, bias=False)
        self.W_k = nn.Linear(in_size, d_model, bias=False)
        self.W_v = nn.Linear(in_size, d_model, bias=False)

        # output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # project to q/k/v and reshape for heads
        q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_head)
        k = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_head)
        v = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_head)

        # transpose to shape: (batch, num_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.score_norm
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)  # (batch, num_heads, seq_len, d_head)

        # combine heads
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(context)
