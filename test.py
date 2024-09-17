from models import *
import torch

# Define parameters
batch_size = 2
seq_len_src = 10
seq_len_tgt = 8
d_model = 16  # Number of features (dimensions)
n_features = 5  # Output features

# Create random source and target sequences with the same batch size
src = torch.randn(batch_size, seq_len_src, d_model)  # Source sequence
tgt = torch.randn(batch_size, seq_len_tgt, d_model)  # Target sequence

# Create the model
model = TransformerModel(
    d_model=d_model, out_features=n_features, n_layers=2, dropout=0.1
)

# Perform a forward pass
output = model(src, tgt)

print("Source Shape:", src.shape)
print("Target Shape:", tgt.shape)
print("Output Shape:", output.shape)
print(output)
