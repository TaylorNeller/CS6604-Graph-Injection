import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear
import torch.nn as nn
import numpy as np

# Load MUTAG dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
dataset = dataset.shuffle()

# Use the test portion of the dataset
_, test_dataset = torch.utils.data.random_split(dataset, [150, len(dataset)-150], 
                                              generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the model architecture (must match the saved model)
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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Try to load JIT model first
        print("Attempting to load JIT model...")
        model = torch.jit.load('best_model_jit.pt')
        print("Successfully loaded JIT model")
    except Exception as e:
        print(f"Failed to load JIT model: {e}")
        print("Attempting to load standard model...")
        
        # Load standard model
        model = EfficientGNN(
            num_features=dataset.num_features,
            hidden_channels=32
        ).to(device)
        model.load_state_dict(torch.load('best_model.pt'))
        print("Successfully loaded standard model")
    
    # Test the model
    model = model.to(device)
    accuracy, predictions = test_model(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print detailed prediction statistics
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples ({count/len(predictions)*100:.2f}%)")

if __name__ == "__main__":
    main()