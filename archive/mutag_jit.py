import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple

# Load MUTAG dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

# Split dataset into train and test sets
torch.manual_seed(42)
dataset = dataset.shuffle()
train_dataset = dataset[:150]
test_dataset = dataset[150:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class EfficientGNN(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # First conv with ReLU
        x = F.relu(self.conv1(x, edge_index))
        
        # Second conv with ReLU
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_add_pool(x, batch)
        
        # Final classification
        return self.lin(x)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
    return correct / total

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(train_loader.dataset)

def prepare_jit_model(model, device):
    """Prepare model for JIT compilation"""
    model.eval()
    
    # Get sample data for tracing
    sample_data = next(iter(train_loader)).to(device)
    
    # Create example inputs
    example_inputs = (
        sample_data.x,
        sample_data.edge_index,
        sample_data.batch
    )
    
    try:
        # Try scripting first
        print("Attempting to script model...")
        scripted_model = torch.jit.script(model)
        print("Successfully scripted model")
        return scripted_model
    except Exception as e:
        print(f"Scripting failed: {e}")
        print("Falling back to tracing...")
        # Fall back to tracing if scripting fails
        traced_model = torch.jit.trace(model, example_inputs)
        print("Successfully traced model")
        return traced_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = EfficientGNN(
        num_features=dataset.num_features,
        hidden_channels=32
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Compile model after initialization
    try:
        print("Compiling model with JIT...")
        model = prepare_jit_model(model, device)
        print("JIT compilation completed")
        use_jit = True
    except Exception as e:
        print(f"JIT compilation failed: {e}")
        print("Continuing with non-JIT model")
        use_jit = False
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10)
    
    # Training loop with progress bar
    pbar = tqdm(range(100))
    best_test_acc = 0
    
    for epoch in pbar:
        # Train
        loss = train_epoch(model, train_loader, optimizer, device)
        
        # Test
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        
        # Update progress bar
        pbar.set_description(f'Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # Early stopping check
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if use_jit:
                torch.jit.save(model, 'best_model_jit.pt')
            else:
                torch.save(model.state_dict(), 'best_model.pt')

    print(f"Best test accuracy: {best_test_acc:.4f}")

if __name__ == '__main__':
    main()