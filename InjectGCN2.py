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
from collections import Counter


from GraphGAN import UnboundAttack
from GCNModel2 import *

n_samples = 100
n_training_epochs = 50

# 50: .6105
# 40: .5526
# 30: .5789
# 20: .5148
# 10: .6053
#  0: .6369


batch_size = 1
latent_dim=100
beta=0
lambda_degree = 300
epochs = 3
epoch_ratio=3

max_nodes = 28

# Load the MUTAG dataset
dataset = TUDataset(root='data/PROTEINS', name='PROTEINS')  

# Shuffle and split the dataset into training and test sets
torch.manual_seed(42)
dataset = dataset.shuffle()
input_dim = dataset.num_features
n_classes = dataset.num_classes
dataset = [data for data in dataset if data.num_nodes <= max_nodes]
dataset = dataset[:188]


train_size = int(len(dataset) * 0.8)
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)




# load model from file
model = GCN(input_dim, 64, n_classes)
model.load_state_dict(torch.load('model/victim_model.pth'))

# load degree distribution
target_degree_dist = np.load('stats/target_degree_dist.npy')  # shape: [max_degree+1]
target_degree_dist = torch.tensor(target_degree_dist, dtype=torch.float32).cuda()

# Initialize attack framework and hyperparameters
attack = UnboundAttack(
    latent_dim=latent_dim,
    num_nodes=28,  # Set to maximum number of nodes in the dataset
    node_features=input_dim,
    victim_model=model,
    target_degree_dist=target_degree_dist,
    device='cuda'
)

# load model
attack.load_models('model')

generated_data_list = []
if (n_samples > 0):
    fake_adj, fake_features = attack.generate_attack(num_samples=n_samples)

    sample_data = train_dataset[0]
    # has_edge_attr = hasattr(sample_data, 'edge_attr')
    has_edge_attr = sample_data.edge_attr is not None
    edge_attr_dim = sample_data.edge_attr.size(1) if has_edge_attr else None

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
        
        # Create a temporary Data object to feed into the model
        temp_data = Data(x=feat_i, edge_index=new_edge_index)
        
        # Predict the label
        with torch.no_grad():
            temp_data = temp_data.to('cuda')  # Ensure the data is on the right device
            temp_data.batch = torch.zeros(temp_data.num_nodes, dtype=torch.long, device='cuda')
            pred = model(temp_data.x, temp_data.edge_index, temp_data.batch)
            predicted_label = pred.argmax(dim=1).item()  # Get the predicted label as an integer
        
        # Assign a label different from the predicted label
        num_classes = n_classes
        different_label = (predicted_label) % num_classes  # Simple scheme to pick a different label
        
        # # Create y tensor with matching shape
        if len(sample_data.y.shape) == 0:
            y = torch.tensor(0)
        else:
            y = torch.zeros_like(sample_data.y)
            y[0] = different_label

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

# Get the labels
labels_train = [data.y.item() for data in train_dataset]
labels_test = [data.y.item() for data in test_dataset]
labels_adv = [data.y.item() for data in generated_data_list]
labels_combined = [data.y.item() for data in combined_dataset]

print("Label Distributions\n\n")
print("Train Dataset:")
train_dist = Counter(labels_train)
for label, count in train_dist.items():
    print(f"Label {label}: {count}")
print("\nTest Dataset:")
test_dist = Counter(labels_test)
for label, count in test_dist.items():
    print(f"Label {label}: {count}")
print("\nAdversarial Dataset:")
adv_dist = Counter(labels_adv)
for label, count in adv_dist.items():
    print(f"Label {label}: {count}")
print("\nCombined Dataset:")
combined_dist = Counter(labels_combined)
for label, count in combined_dist.items():
    print(f"Label {label}: {count}")

trainer = GCNTrainer()
trainer.train_loader = combined_loader
trainer.test_loader = test_loader
trainer.train_eval(n_training_epochs)
trainer.save_model('model/victim_model_adv.pth')

