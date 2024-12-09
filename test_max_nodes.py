import torch
from torch_geometric.datasets import TUDataset

# Load the MUTAG dataset
dataset = TUDataset(root='data/MUTAG', name='MUTAG')

# Initialize variables to track the largest graph
max_nodes = 0
largest_graph = None

# Iterate through the dataset to find the graph with the most nodes
for data in dataset:
    num_nodes = data.num_nodes
    if num_nodes > max_nodes:
        max_nodes = num_nodes
        largest_graph = data

print(f'The largest graph has {max_nodes} nodes.')
