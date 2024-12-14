# Imports, as always...

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# The baseline GCN.
class GCN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hid_dim: int,
            n_classes: int,
            n_layers: int,
            dropout_ratio: float = 0.1
    ):

        super(GCN, self).__init__()
        '''
        Args:
            input_dim: input feature dimension
            hid_dim: hidden feature dimension
            n_classes: number of target classes
            n_layers: number of layers
            dropout_ratio: dropout ratio (for training)
        '''

        # Parameters.
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio

        # Special case for no layers.
        self.layers = nn.ModuleList([nn.Linear(input_dim, n_classes)])

        if n_layers and n_layers > 0:
            # Layer dimensions.
            input_dims = [input_dim] + ([hid_dim] * (n_layers - 1))
            output_dims = ([hid_dim] * (n_layers - 1)) + [n_classes]

            # List of layers.
            self.layers = nn.ModuleList([
                GCNConv(x, y) for x, y in zip(input_dims, output_dims)
            ])

        # Activation function.
        self.activation = F.relu

    def forward(self, X, A) -> torch.Tensor:
        if self.n_layers and self.n_layers != 0:
            for layer in self.layers[:-1]:
                # Update the node features, then activate.
                X = layer(X, A)
                X = self.activation(X)

                # Dropout during training.
                X = F.dropout(X, p=self.dropout_ratio, training=self.training)

            # No activation (or dropout) on the final layer.
            X = self.layers[-1](X, A)

        # Special case for 0-layer model.
        else:
            X = self.layers[0](X)

        return X

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        return F.log_softmax(self.forward(X, A), dim=1)