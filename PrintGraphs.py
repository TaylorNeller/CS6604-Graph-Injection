import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import is_undirected


from GraphGAN import UnboundAttack
from GCNModel import *

n_samples = 30

batch_size = 1
latent_dim=100
beta=0
lambda_degree = 300
epochs = 3
epoch_ratio=3


# Load the MUTAG dataset
dataset = TUDataset(root='data/MUTAG', name='MUTAG')

# Shuffle and split the dataset into training and test sets
torch.manual_seed(42)
dataset = dataset.shuffle()
train_size = int(len(dataset) * 0.8)
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)




# Initialize the model, optimizer, and loss function
input_dim = dataset.num_features

# load model from file
model = GCN(input_dim, 64, dataset.num_classes)
model.load_state_dict(torch.load('model/victim_model.pth'))

# load degree distribution
target_degree_dist = np.load('stats/target_degree_dist.npy')  # shape: [max_degree+1]
target_degree_dist = torch.tensor(target_degree_dist, dtype=torch.float32).cuda()

# Initialize attack framework and hyperparameters
attack = UnboundAttack(
    latent_dim=latent_dim,
    num_nodes=28,  # Set to maximum number of nodes in the dataset
    node_features=input_dim,  # dataset.num_features ensures consistency
    victim_model=model,
    target_degree_dist=target_degree_dist,
    device='cuda'
)

# load model
attack.load_models('model')

fake_adj, fake_features = attack.generate_attack(num_samples=n_samples)

def visualize_adjacency_matrix(tensor, f_name):
    """
    Visualize the first adjacency matrix in the given tensor.
    
    Parameters:
        tensor (numpy.ndarray): A 3D numpy array where the first dimension indexes the adjacency matrices.
    """
    first_matrix = tensor
    if tensor.ndim == 3:    
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
    
        first_matrix = tensor[0]  # Extract the first adjacency matrix
        
        # Check if the matrix is square
        if first_matrix.shape[0] != first_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        
        # convert any >= .5 values to 1
        first_matrix[first_matrix >= 0.5] = 1

        # Remove empty rows/cols
        non_empty_rows = np.any(first_matrix != 0, axis=1)
        non_empty_cols = np.any(first_matrix != 0, axis=0)
        first_matrix = first_matrix[non_empty_rows][:, non_empty_cols]


    else:
        first_matrix = first_matrix.cpu().numpy()

    # Create a graph from the adjacency matrix
    graph = nx.from_numpy_array(first_matrix)
    
    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(graph, with_labels=False, node_color="lightblue", node_size=500, font_size=10, font_weight="bold", edge_color="gray")
    plt.title("Graph Visualization of the First Adjacency Matrix")
    plt.savefig(f_name, format="png", dpi=300)
    plt.close()

# Call the function with t_adj
for i in range(len(fake_adj)):
    visualize_adjacency_matrix(fake_adj, f'graphs/graph_vis_fake_{i}.png')
    fake_adj = fake_adj[1:]