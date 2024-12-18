# Imports, as always...

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_kmeans import KMeans, CosineSimilarity

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# A backbone model that will accommodate the different components.
class BackboneModel(nn.Module):
    def __init__(
            self,
            residual_method: nn.Module,
            aggregation_function: nn.Module,
            n_nodes: int,
            input_dim: int,
            hid_dim: int,
            n_classes: int,
            n_layers: int,
            K: int = 10,
            L_0: int = 1,
            dropout_ratio: float = 0.1,
            act_fn=F.relu,
    ):

        super(BackboneModel, self).__init__()
        '''
        Args:
            residual_method: nn.Module to use as the method for learning residual weighted connections
            aggregation_function: arbitrary aggregation function of a GNN model
            n_nodes: number of nodes in the graph
            input_dim: input feature dimension
            hid_dim: hidden feature dimension
            n_classes: number of target classes
            n_layers: number of layers
            K: number of clusters for K-means clustering (only used for the cluster-keeping residual method)
            L_0: threshold layer to begin residual connections
            dropout_ratio: dropout ratio (for training)
            act_fn: nonlinear activation function to use
        '''

        # Parameters.
        self.residual_method = residual_method
        self.aggregation_function = aggregation_function
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio
        self.act_fn = act_fn

        # For cluster-keeping, having persistent access to the L_0-th node embeddings is necessary.
        self.K = K
        self.L_0 = L_0

        # Special case for no layers.
        self.layers = nn.ModuleList([nn.Linear(input_dim, n_classes)])

        if n_layers and n_layers > 0:
            # Layer dimensions.
            input_dims = [input_dim] + ([hid_dim] * (n_layers - 1))
            output_dims = ([hid_dim] * (n_layers - 1)) + [n_classes]

            # List of layers.
            self.layers = nn.ModuleList([
                # Only adding residual connections between layers. Only GCNConv offers normalisation, so leave it.
                residual_method(n_nodes, aggregation_function(x, y, add_self_loops=True), act_fn)
                if x == y and residual_method and l >= L_0 else aggregation_function(x, y)
                for l, (x, y) in enumerate(zip(input_dims, output_dims))
            ])

    def forward(self, X, A) -> torch.Tensor:
        if self.n_layers and self.n_layers != 0:
            for l, layer in enumerate(self.layers[:-1]):
                # Once we hit layer L_0, cluster and send the result to the layers that need it.
                # Check the residual method -- let's not waste resources if we don't have to!
                # Also, this is only something we should bother with during training.
                if l == self.L_0 and self.residual_method is ClusterKeepingRC and self.training:
                    for cluster_keeping_layer in self.layers[self.L_0:-1]:
                        # Passing raw logits because a log softmax is just not working out.
                        cluster_keeping_layer.update_cluster_info(X)

                # Update the node features.
                X = layer(X, A)

                # Activation.
                X = self.act_fn(X)

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


# A baseline GCN. This should be equivalent to the BackboneModel class with residual_method = None.
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


# Fixing the residual connections (given).
class FixedRC(nn.Module):
    def __init__(
            self,
            fixed_weights: torch.Tensor,
            n_nodes: int,
            aggregation_function: nn.Module,
            act_fn=F.relu
    ):
        super(FixedRC, self).__init__()
        '''
        Args:
            fixed_weights: weights to use for residual connections
            n_nodes: not used -- it simplifies the implementation of the Backbone to leave this be.
            aggregation_function: arbitrary aggregation function of a GNN model
            act_fn: nonlinear activation function
        '''

        # In accordance with the notation of Liu et al. (2024).
        self.f = aggregation_function
        self.sigma = act_fn

        # Aggregation weights (as a vector that will be diagonalised into a matrix in the forward).
        # These should be moved to the correct device prior to instantiation of the class.
        self.w = nn.Parameter(fixed_weights, requires_grad=False)
        self.I = nn.Parameter(torch.eye(fixed_weights.size(0)), requires_grad=False)

    def forward(self, X, A_hat) -> torch.Tensor:
        # Diagonalise the weights into a matrix.
        xi = torch.diag_embed(self.w)

        # Compute the two terms.
        aggregated_term = torch.matmul(xi, self.sigma(self.f(X, A_hat)))
        residual_term = torch.matmul(self.I - xi, X)

        # Return their sum.
        return aggregated_term + residual_term

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        return F.log_softmax(self.forward(X, A), dim=1)


# Freely learning the residual connections.
class FreeRC(nn.Module):
    def __init__(
            self,
            n_nodes: int,
            aggregation_function: nn.Module,
            act_fn = F.relu
    ):
        super(FreeRC, self).__init__()
        '''
        Args:
            n_nodes: number of nodes in the graph
            aggregation_function: arbitrary aggregation function of a GNN model
            act_fn: nonlinear activation function
        '''

        # In accordance with the notation of Liu et al. (2024).
        self.f = aggregation_function
        self.sigma = act_fn

        # Aggregation weights (as a vector that will be diagonalised into a matrix in the forward).
        self.n_nodes = n_nodes
        self.w = nn.Parameter(torch.rand(n_nodes), requires_grad=True)
        self.I = nn.Parameter(torch.eye(n_nodes), requires_grad=False)

    def forward(self, X, A_hat) -> torch.Tensor:
        # Diagonalise the weights into a matrix (this is differentiable).
        # Also, pass xi through a sigmoid function to clamp the values to [0,1].
        xi = torch.diag_embed(torch.sigmoid(self.w))

        # Compute the two terms.
        aggregated_term = torch.matmul(xi, self.sigma(self.f(X, A_hat)))
        residual_term = torch.matmul(self.I - xi, X)

        # Return their sum.
        return aggregated_term + residual_term

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        return F.log_softmax(self.forward(X, A), dim=1)


class HierarchicalRC(nn.Module):
    def __init__(
            self,
            n_nodes: int,
            aggregation_function: nn.Module,
            act_fn = F.relu
    ):
        super(HierarchicalRC, self).__init__()
        '''
        Args:
            fixed_weights: weights to use for residual connections
            n_nodes: number of nodes in the graph
            aggregation_function: arbitrary aggregation function of a GNN model
            act_fn: nonlinear activation function
        '''

        # In accordance with the notation of Liu et al. (2024).
        self.f = aggregation_function
        self.sigma = act_fn

        # Aggregation weights (as a vector that will be diagonalised into a matrix in the forward).
        # These should be moved to the correct device prior to instantiation of the class.
        self.w_global = nn.Parameter(torch.rand(1), requires_grad=True)
        self.w_local = nn.Parameter(torch.rand(n_nodes), requires_grad=True)
        self.I = nn.Parameter(torch.eye(n_nodes), requires_grad=False)

    def forward(self, X, A_hat) -> torch.Tensor:
        # Compute the node weights from the combination of global and local weights.
        # All weights clamped to [0,1] via a sigmoid.
        xi = torch.sigmoid(self.w_global) * torch.diag_embed(torch.sigmoid(self.w_local))

        # Compute the two terms.
        aggregated_term = torch.matmul(xi, self.sigma(self.f(X, A_hat)))
        residual_term = torch.matmul(self.I - xi, X)

        # Return their sum.
        return aggregated_term + residual_term

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        return F.log_softmax(self.forward(X, A), dim=1)


# Freely learning the residual connections.
class ClusterKeepingRC(nn.Module):
    def __init__(
            self,
            n_nodes: int,
            aggregation_function: nn.Module,
            act_fn=F.relu,
            # These hyperparameters are not set by the Backbone model -- do these manually.
            alpha: float = 1.,
            K: int = 5
    ):
        super(ClusterKeepingRC, self).__init__()
        '''
        Args:
            n_nodes: number of nodes in the graph
            aggregation_function: arbitrary aggregation function of a GNN model
            alpha: hyperparameter in [0, 1] determining the influence of the global control
            act_fn: nonlinear activation function
        '''

        # In accordance with the notation of Liu et al. (2024).
        self.f = aggregation_function
        self.sigma = act_fn

        # Aggregation weights (as a vector that will be diagonalised into a matrix in the forward).
        self.n_nodes = n_nodes
        self.theta = torch.tensor(0).to(device)
        self.I = torch.eye(n_nodes).to(device)

        # Hyperparameters.
        self.K = K
        self.alpha = alpha

        # Keeping the cluster information of the L_0-th layer.
        self.cluster_ids_0 = None
        self.cluster_centers_0 = None
        self.dist_0 = None
        self.dist_l = None

    def forward(self, X, A_hat) -> torch.Tensor:
        # Recalculate the control parameters.
        # This should only happen during training -- it's not cheap!
        if self.training:
            # We're going to pass raw logits, because a log softmax puts roughly equates all features.
            self.theta = self.compute_theta(X)

        # Diagonalise the control parameters into a matrix (this is differentiable).
        xi = torch.diag_embed(torch.tensor([self.theta] * self.n_nodes)).to(torch.float).to(device)

        # Compute the two terms.
        aggregated_term = torch.matmul(xi, self.sigma(self.f(X, A_hat)))
        residual_term = torch.matmul(self.I - xi, X)

        # Return their sum.
        return aggregated_term + residual_term

    # Computing the objective of the K-means.
    def compute_dist(self, H, cluster_ids, cluster_centers):
        # We use mean rather than sum to avoid enormous distance values.
        return torch.mean(torch.linalg.norm(H - cluster_centers[cluster_ids], dim=1))

    def update_cluster_info(self, H_0):
        # Update the info.
        cluster_info = self.kmeans_cluster_nodes(H_0)
        self.cluster_ids_0, self.cluster_centers_0 = cluster_info.labels[0], cluster_info.centers[0]

        # Recalculate dist_0 with the newly updated info.
        self.dist_0 = self.compute_dist(H_0, self.cluster_ids_0, self.cluster_centers_0)

    # Computing the global control parameter.
    def compute_theta(self, H):
        with torch.no_grad():
            clustering = self.kmeans_cluster_nodes(H)
            self.dist_l = self.compute_dist(H, clustering.labels[0], clustering.centers[0])

            return torch.tensor(min((self.dist_0 / self.dist_l) ** self.alpha, 1)).to(device)

    def kmeans_cluster_nodes(self, H):
        # Cluster the nodes for the l-th layer.
        kmeans = KMeans(
            n_clusters=self.K,
            #init_method='k-means++',
            normalize='unit',
            verbose=False
        ).to(device)

        return kmeans(H.unsqueeze(0))