import torch
from collections import deque


class MemoryBuffer:
    def __init__(self, maxlen: int, mem_dim: int):
        self.mem_dim = mem_dim
        self.buffer = deque(maxlen=maxlen)

    def append(self, encoding: torch.Tensor):
        self.buffer.append(encoding)

    def read_mem(self):
        if len(self.buffer) == 0:
            return torch.zeros(1, 1, self.mem_dim)
        return torch.cat(list(self.buffer), dim=0).unsqueeze(0)
