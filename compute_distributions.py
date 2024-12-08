import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import os

# Parameters (ensure these match what's used in your training code)
DATA_ROOT = 'data/MUTAG'
DATASET_NAME = 'MUTAG'
MAX_NUM_NODES = 28  # same as used in the generator and victim code
CLUST_BINS = np.linspace(0, 1, 11)  # 10 bins for clustering coefficient

# Load dataset
dataset = TUDataset(root=DATA_ROOT, name=DATASET_NAME)
dataset = dataset.shuffle()
train_size = int(len(dataset) * 1)
train_dataset = dataset[:train_size]

# Create a loader for the entire training dataset
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# We'll collect global statistics over all training graphs
all_degrees = []
all_clust_vals = []

degree_zero_count = 0  # Counter for degree-0 nodes


for data in train_loader:
    # data.x: [num_nodes, num_features]
    # data.edge_index: [2, num_edges]
    # Convert to dense adjacency
    adj = to_dense_adj(data.edge_index, max_num_nodes=MAX_NUM_NODES).squeeze(0).numpy()
    # There might be padding if the graph is smaller than MAX_NUM_NODES.
    # We can remove rows and columns that correspond to no nodes if needed.
    # However, since data is from TUDataset, each data in a batch of 1 is just one graph
    # If there's zero-padding, it should be isolated as empty rows/cols.

    # Identify actual number of nodes (non-padded)
    # A heuristic: node i exists if it has a feature row in data.x
    num_nodes = data.x.size(0)
    adj = adj[:num_nodes, :num_nodes]

    # Compute degrees
    degrees = adj.sum(axis=1)
    all_degrees.append(degrees)

    # Count degree-0 nodes due to padding
    degree_zero_count += MAX_NUM_NODES - num_nodes

    # Compute clustering coefficients using NetworkX
    G = nx.from_numpy_array(adj)
    clust_dict = nx.clustering(G)
    clust_values = np.array(list(clust_dict.values()))
    all_clust_vals.append(clust_values)

# Concatenate all degree and clustering values
all_degrees = np.concatenate(all_degrees)  # all node degrees from all training graphs
all_clust_vals = np.concatenate(all_clust_vals)  # all node clustering coefficients

# Compute global degree distribution
# Maximum degree can be up to num_nodes-1 in a fully connected graph
max_degree = MAX_NUM_NODES - 1
degree_counts = np.bincount(all_degrees.astype(int), minlength=max_degree+1)
degree_counts[0] += degree_zero_count  # Include degree-0 nodes
degree_dist = degree_counts / degree_counts.sum()  # Normalize

# Compute global clustering distribution
# Use the same bins as in training code
clust_hist, _ = np.histogram(all_clust_vals, bins=CLUST_BINS, density=True)

# Create a directory to save the distributions if needed
os.makedirs('precomputed_stats', exist_ok=True)

# Save the distributions as .npy files
np.save('stats/target_degree_dist.npy', degree_dist)
np.save('stats/target_clust_dist.npy', clust_hist)

print("Saved target distributions:")
print("Degree distribution:", degree_dist)
print("Clustering distribution:", clust_hist)
print("Files saved to stats/target_degree_dist.npy and stats/target_clust_dist.npy")
