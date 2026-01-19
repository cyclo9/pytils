import torch


def encode_position(x: torch.Tensor):
    d_model = x.shape[2]
    device = x.device

    p = torch.arange(x.shape[1]).to(device).unsqueeze(1)
    i = torch.arange(d_model).to(device).unsqueeze(0)

    frequency = 1 / torch.pow(10000, (2 * (i // 2) / d_model))
    freq_rads = p * frequency

    PE = torch.zeros_like(freq_rads)
    PE[:, 0::2] = torch.sin(freq_rads[:, 0::2])
    PE[:, 1::2] = torch.cos(freq_rads[:, 1::2])

    return x + PE.unsqueeze(0)
