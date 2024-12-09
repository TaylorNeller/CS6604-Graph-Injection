from torch_geometric.datasets import TUDataset
from collections import Counter

# Load the MUTAG dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

# Get the labels
labels = [data.y.item() for data in dataset]

# Calculate label distribution
label_distribution = Counter(labels)

# Print label distribution
print("Label Distribution:")
for label, count in label_distribution.items():
    print(f"Label {label}: {count}")
