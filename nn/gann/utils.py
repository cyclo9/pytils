import torch


def calc_num_genes(in_size, out_size, hidden_layers, n_units):
    num_genes = (
        ((in_size * n_units) + n_units)
        + (hidden_layers * ((n_units**2) + n_units))
        + ((out_size * n_units) + out_size)
    )
    return num_genes


def calc_num_genes_lstm(in_size, out_size, hidden_layers, n_units):
    num_genes = (
        4 * ((in_size + n_units) * n_units + n_units)
        + hidden_layers * 4 * ((n_units**2) + n_units)
        + 4 * ((out_size * n_units) + out_size)
    )
    return num_genes


def reload_model(model, params):
    """Reload a model given a list of weights and bias. Assumes the params aren't already tensors."""

    start_idx = 0
    params = torch.tensor(params, dtype=torch.float32)
    for param in model.parameters():
        # Get the number of elements in the parameter (weights or biases)
        param_size = param.numel()

        # Assign the appropriate slice from params
        param.data = params[start_idx : start_idx + param_size].view(
            param.shape
        )
        start_idx += param_size

    return model
