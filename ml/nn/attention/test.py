import torch

from main import AttentionBlock

in_size = 4
seq_len = 2
block = AttentionBlock(4, 12, 3)

x = torch.randn(3, seq_len, in_size)

out = block(x)
print(x.shape)
print(out.shape)
