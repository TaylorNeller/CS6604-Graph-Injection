import torch
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt

# Load the PROTEINS dataset
dataset = TUDataset(root="./data/PROTEINS", name="PROTEINS")

# Calculate the number of nodes in each graph
num_nodes_list = [data.num_nodes for data in dataset]
# find how many graphs have <= 28 nodes
num_nodes_28 = len([x for x in num_nodes_list if x <= 28])
print(f'Number of graphs with <= 28 nodes: {num_nodes_28}')

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(num_nodes_list, bins=20, edgecolor='k', alpha=0.7)
plt.title("Distribution of Number of Nodes in Graphs (PROTEINS Dataset)")
plt.xlabel("Number of Nodes")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.savefig('node_dist', format="png", dpi=300)