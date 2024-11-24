import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Module
import torch.nn as nn
from typing import Optional, Tuple

# Load MUTAG dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

# Use the test portion of the dataset
_, test_dataset = torch.utils.data.random_split(dataset, [150, len(dataset)-150], 
                                              generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=64)

# Original model architecture (for loading weights)
class EfficientGNN(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        return self.lin(x)

# TorchScript compatible model
class ScriptableGCN(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)
        
    def normalize_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        norm = self.normalize_adj(edge_index, num_nodes)
        out = torch.zeros_like(x)
        msg = x[col] * norm.view(-1, 1)
        out.scatter_add_(0, row.view(-1, 1).expand(-1, x.size(1)), msg)
        return self.linear(out)

class TorchScriptGNN(Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = ScriptableGCN(num_features, hidden_channels)
        self.conv2 = ScriptableGCN(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
    
    def global_add_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        dim_size = int(batch.max().item() + 1)
        return torch.zeros(dim_size, x.size(-1), device=x.device).scatter_add_(
            0, batch.view(-1, 1).expand(-1, x.size(-1)), x)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.global_add_pool(x, batch)
        return self.lin(x)

def transfer_weights(old_model, new_model):
    """Transfer weights from old model to new TorchScript-compatible model"""
    # Transfer conv1 weights
    new_model.conv1.linear.weight.data = old_model.conv1.lin.weight.data
    new_model.conv1.linear.bias.data = old_model.conv1.bias.data
    
    # Transfer conv2 weights
    new_model.conv2.linear.weight.data = old_model.conv2.lin.weight.data
    new_model.conv2.linear.bias.data = old_model.conv2.bias.data
    
    # Transfer final linear layer weights
    new_model.lin.weight.data = old_model.lin.weight.data
    new_model.lin.bias.data = old_model.lin.bias.data
    
    return new_model

@torch.no_grad()
def test_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
        predictions.extend(pred.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the original model
    old_model = EfficientGNN(
        num_features=dataset.num_features,
        hidden_channels=32
    ).to(device)
    
    # Initialize the new model
    new_model = TorchScriptGNN(
        num_features=dataset.num_features,
        hidden_channels=32
    ).to(device)
    
    try:
        print("Loading original model state...")
        old_model.load_state_dict(torch.load('best_model.pt'))
        
        print("Transferring weights to new model...")
        new_model = transfer_weights(old_model, new_model)
        
        print("Converting to TorchScript...")
        scripted_model = torch.jit.script(new_model)
        print("Successfully converted to TorchScript")
        
        # Save the scripted model
        print("Saving TorchScript model...")
        torch.jit.save(scripted_model, 'scripted_model.pt')
        
        # Test both models to verify transfer
        print("\nTesting original model...")
        old_accuracy, old_predictions = test_model(old_model, test_loader, device)
        print(f"Original model accuracy: {old_accuracy:.4f}")
        
        print("\nTesting scripted model...")
        new_accuracy, new_predictions = test_model(scripted_model, test_loader, device)
        print(f"Scripted model accuracy: {new_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error during model transfer/testing: {e}")
        return
    
    print(f"\nTest Results:")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Original model accuracy: {old_accuracy:.4f}")
    print(f"Scripted model accuracy: {new_accuracy:.4f}")
    
    # Print detailed prediction statistics
    import numpy as np
    unique, counts = np.unique(new_predictions, return_counts=True)
    print("\nPrediction distribution (scripted model):")
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples ({count/len(new_predictions)*100:.2f}%)")

if __name__ == "__main__":
    main()