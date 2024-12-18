# Imports, as always...

import typing

# PyTorch (+ Geometric)
import torch
import torch.nn.functional as F

import torch_geometric

# Notebook imports.
from tqdm.notebook import tqdm


# Generic training loop.
def train(
        model: torch.nn.Module,
        data: torch_geometric.data.data.Data,
        device: str,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        max_patience: int = 10,
        progress_bar: bool = True,
        verbose: bool = True,
        print_interval: int = 10
) -> typing.Dict:
    # Move to device.
    model.to(device)
    data.to(device)

    # Loss function.
    loss_fn = F.cross_entropy

    # Optimiser.
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Tracking the statistics as we go.
    stats = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
    }

    # Main loop.
    for epoch_idx in (tqdm(range(1, epochs + 1), desc='Training') if progress_bar else range(1, epochs + 1)):
        # Pop it into train mode and zero gradients.
        model.train()
        optimiser.zero_grad()

        # No batches -- just the single (big) graph in each epoch.
        logits = model(data.x, data.edge_index)

        # Compute loss (masked for training nodes only).
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

        # Gradient calculation and weight update.
        loss.backward()
        optimiser.step()

        # And into evaluation mode, and ignore gradients.
        model.eval()
        with torch.no_grad():
            # Get the logits and node embeddings.
            logits = model(data.x, data.edge_index)
            node_embeddings = F.log_softmax(logits)

            # Compute train loss and accuracy (percentage of nodes classified correctly).
            train_loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            train_acc = torch.mean(
                (torch.argmax(node_embeddings[data.train_mask], dim=1) == data.y[data.train_mask]).to(float))

            stats['train_loss'].append(train_loss.item())
            stats['train_acc'].append(train_acc.item())

            # And again for val.
            val_loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_acc = torch.mean(
                (torch.argmax(node_embeddings[data.val_mask], dim=1) == data.y[data.val_mask]).to(float))

            stats['val_loss'].append(val_loss.item())
            stats['val_acc'].append(val_acc.item())

            if verbose and epoch_idx % print_interval == 0:
                print(
                    f'Epoch {epoch_idx:03d}: train loss - {train_loss:.3f}, train acc - {train_acc:.3f}, val loss - {val_loss:.3f}, val acc - {val_acc:.3f}')

            # TODO: Early stopping if necessary.

    return stats


# Evaluating on the test set.
def evaluate(
        model: torch.nn.Module,
        data: torch_geometric.data.data.Data,
        device: str
) -> typing.Tuple:
    # Move to device.
    model.to(device)
    data.to(device)

    # Loss function.
    loss_fn = F.cross_entropy

    # Into evaluation mode, and ignore gradients.
    model.eval()
    with torch.no_grad():
        # Get the logits and node embeddings.
        logits = model(data.x, data.edge_index)
        node_embeddings = model.generate_node_embeddings(data.x, data.edge_index)

        # Compute test loss and accuracy.
        loss = loss_fn(logits[data.mask], data.y[data.mask])
        acc = torch.mean((torch.argmax(node_embeddings[data.mask], dim=1) == data.y[data.mask]).to(float))

        return loss, acc
