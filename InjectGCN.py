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
from victim import *


# Load the MUTAG dataset
dataset = TUDataset(root='data/MUTAG', name='MUTAG')

# Shuffle and split the dataset into training and test sets
torch.manual_seed(42)
dataset = dataset.shuffle()
train_size = int(len(dataset) * 0.8)
train_dataset = dataset[:train_size]
# test_dataset = dataset[train_size:]
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
    latent_dim=100,
    num_nodes=28,  # Set to maximum number of nodes in the dataset
    node_features=input_dim,  # dataset.num_features ensures consistency
    victim_model=model,
    target_degree_dist=target_degree_dist,
    device='cuda',
    beta=0.1,
    lambda_degree = 1000.0,
)

# load model
attack.load_models('model')

fake_adj, fake_features = attack.generate_attack(num_samples=10, target_class=0)

sample_data = train_dataset[0]
has_edge_attr = hasattr(sample_data, 'edge_attr')
edge_attr_dim = sample_data.edge_attr.size(1) if has_edge_attr else None

generated_data_list = []
num_samples = fake_adj.shape[0]
sample_y_shape = train_dataset[0].y.shape
for i in range(num_samples):
    adj_i = fake_adj[i]
    feat_i = fake_features[i]

    # Ensure correct device and type
    adj_i = adj_i.cpu() if adj_i.is_cuda else adj_i
    feat_i = feat_i.cpu() if feat_i.is_cuda else feat_i
    
    # Convert dense adjacency to edge_index
    edge_index, _ = dense_to_sparse(adj_i)
    
    # Remove any unused nodes (nodes with no edges)
    used_nodes = torch.unique(edge_index)
    feat_i = feat_i[used_nodes]
    
    # Remap edge indices to account for removed nodes
    node_idx_map = {int(old): new for new, old in enumerate(used_nodes)}
    new_edge_index = torch.tensor([[node_idx_map[int(edge_index[0, i])], 
                                   node_idx_map[int(edge_index[1, i])]] 
                                  for i in range(edge_index.shape[1])]).t()
    
    # Create y tensor with matching shape
    if len(sample_data.y.shape) == 0:
        y = torch.tensor(0)
    else:
        y = torch.zeros_like(sample_data.y)
        y[0] = 0
    
    # Create edge_attr if needed
    if has_edge_attr:
        num_edges = new_edge_index.size(1)
        edge_attr = torch.ones((num_edges, edge_attr_dim))
    else:
        edge_attr = None
    
    # Create the Data object with matching structure
    data_i = Data(
        x=feat_i,
        edge_index=new_edge_index,
        y=y,
        edge_attr=edge_attr if has_edge_attr else None
    )
    
    if data_i.validate():
        generated_data_list.append(data_i)
    else:
        print(f"Invalid data generated for sample {i}.")
        break

# combine the original and generated data
original_data_list = [train_dataset[i] for i in range(len(train_dataset))]
combined_dataset = original_data_list + generated_data_list
# combined_dataset = [data.to('cuda') for data in combined_dataset]
combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
print("Original dataset size:", len(original_data_list))
print("Combined dataset size:", len(combined_dataset))

trainer = GCNTrainer()
trainer.train_loader = combined_loader
trainer.train_eval(20)
trainer.save_model('model/victim_model_adv.pth')
