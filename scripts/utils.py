# Imports, as always...
import typing

import matplotlib.figure
import torch
import torch_geometric

# Dimensionality reduction.
import pandas as pd
from sklearn.manifold import TSNE

# Visualisation.
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering
from torch_kmeans import KMeans

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Beautification.
sns.set_context('paper')
sns.set_style('darkgrid')
sns.set_palette('Set2')


def kmeans_cluster_nodes(H, K):
    # Cluster the nodes for the l-th layer.
    kmeans = KMeans(
        n_clusters=K,
        init_method='k-means++',
        normalize='unit',
        verbose=False
    ).to(device)

    return kmeans(H.unsqueeze(0))


def produce_reduced_embeddings(
        model: torch.nn.Module,
        data: torch_geometric.data.data.Data,
        mask: torch.Tensor
) -> pd.DataFrame:
    # Into evaluation mode.
    model.eval()

    # Generate node embeddings (as a numpy array).
    embeddings = model.generate_node_embeddings(data.x, data.edge_index)[mask]
    embeddings = embeddings.detach().cpu().numpy()

    # Reduction.
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Return as a dataframe (with associated labels).
    return pd.DataFrame({
        'dimension 1': reduced_embeddings[:, 0],
        'dimension 2': reduced_embeddings[:, 1],
        'labels': data.y[mask].detach().cpu().numpy()
    })


def plot_reduced_embeddings(
    reduced_features_by_model: pd.DataFrame
) -> typing.Tuple:
    keys = list(reduced_features_by_model.keys())

    # Plot onto a row of axes.
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3), sharey='all',
                             sharex='all')
    for i, ax in enumerate(axes):
        sns.scatterplot(
            # Plot points and colour them.
            x=reduced_features_by_model[keys[i]]['dimension 1'],
            y=reduced_features_by_model[keys[i]]['dimension 2'],
            hue=reduced_features_by_model[keys[i]]['labels'],
            ax=ax,

            # Beautification.
            palette='Set2',
            linestyle='--',
            edgecolors='k'
        )

        # Remove clutter.
        ax.legend([], [], frameon=False)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Add titles.
        ax.set_title(f'Depth-{keys[i]}')

    # No need for a legend -- the classes are abstract at this level anyway.
    # axes[-1].legend(
    #    bbox_to_anchor=(1, .5),
    #    loc='center left',
    #    title='Class'
    # )

    return fig, axes
