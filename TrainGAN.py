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
from torch_geometric.datasets import QM9


from GraphGAN import UnboundAttack
from GCNModel import *

batch_size = 1
latent_dim = 100
lambda_degree = 1
beta = 0
epochs = 1
n_gen_epochs = 1
n_critic_epochs = 1
lambda_ = 5
temperature = .5

# Load the MUTAG dataset
dataset = TUDataset(root='data/MUTAG', name='MUTAG')
max_nodes = 28
target_degree_dist = np.load('stats/target_degree_dist.npy')  # shape: [max_degree+1]

# Shuffle and split the dataset into training and test sets
torch.manual_seed(45)
dataset = dataset.shuffle()
train_size = int(len(dataset) * 0.1)
print('train_size:', train_size)
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
input_dim = dataset.num_features

# load model from file
model = GCN(input_dim, 64, dataset.num_classes)
model.load_state_dict(torch.load('model/victim_model.pth'))

# load degree distribution
target_degree_dist = torch.tensor(target_degree_dist, dtype=torch.float32).cuda()

# Initialize attack framework and hyperparameters
attack = UnboundAttack(
    latent_dim=latent_dim,
    num_nodes=max_nodes,  # Set to maximum number of nodes in the dataset
    node_features=input_dim,  # dataset.num_features ensures consistency
    victim_model=model,
    target_degree_dist=target_degree_dist,
    device='cuda',
    beta=beta,
    lambda_degree=lambda_degree,
    n_gen=n_gen_epochs,
    n_critic=n_critic_epochs,
    lambda_=lambda_,
    temperature=temperature
)

# load model
# attack.load_models('model')

# # Training loop
for epoch in range(epochs):
    for batch in train_loader:
        batch = batch.to('cuda')
        true_class = batch.y[0].item()
        d_loss, g_loss = attack.train_step(batch, target_class=true_class)
        # d_loss, g_loss = attack.train_step(batch)
        print(f'Epoch {epoch}: D_loss {d_loss}, G_loss {g_loss}')

# # Save the trained generator
attack.save_models('model')

# Generate adversarial examples
# tensor format
fake_adj, fake_features = attack.generate_attack(num_samples=10)
# print("Fake Examples:")
# print("Adjacency matrices:", fake_adj)
# print("Node features:", fake_features)

real_adj = None
# Print real examples from the dataset
print("\nReal Examples:")
for batch in train_loader:
    # Take just the first few examples from the batch
    for i in range(min(10, batch.num_graphs)):
        # Get the node features for this graph
        mask = batch.batch == i
        nodes = batch.x[mask]
        
        # Get the edge indices for this graph
        edge_mask = torch.logical_and(batch.batch[batch.edge_index[0]] == i,
                                      batch.batch[batch.edge_index[1]] == i)
        edges = batch.edge_index[:, edge_mask]
        
        # Shift edge indices to be relative to this graph
        node_offset = (batch.batch < i).sum()
        edges = edges - node_offset.view(1, -1)
        
        # Create adjacency matrix from edge indices
        n = mask.sum()
        adj = torch.zeros((n, n))
        adj[edges[0], edges[1]] = 1
        
        real_adj = adj
        # print(f"\nGraph {i}:")
        # print("Adjacency matrix:\n", adj)
        # print("Node features:\n", nodes)
    
    # Break after first batch to only show a few examples
    break

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
visualize_adjacency_matrix(fake_adj, 'graph_vis_fake1.png')
fake_adj = fake_adj[1:]
visualize_adjacency_matrix(fake_adj, 'graph_vis_fake2.png')
fake_adj = fake_adj[1:]
visualize_adjacency_matrix(fake_adj, 'graph_vis_fake3.png')

visualize_adjacency_matrix(real_adj, 'graph_vis_real.png')
