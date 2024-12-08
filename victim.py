import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        # Apply graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Apply global mean pooling
        x = global_mean_pool(x, batch)
        
        # Apply the final linear layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GCNTrainer():
    def __init__(self, model=None, optimizer=None, criterion=None, train_loader=None, test_loader=None):
        if model is None:
            # Load the MUTAG dataset
            dataset = TUDataset(root='data/MUTAG', name='MUTAG')

            # Shuffle and split the dataset into training and test sets
            torch.manual_seed(42)
            dataset = dataset.shuffle()
            train_size = int(len(dataset) * 0.8)
            train_dataset = [data.to('cuda') for data in dataset[:train_size]]
            test_dataset = [data.to('cuda') for data in dataset[train_size:]]
            # Create data loaders
            self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Initialize the model, optimizer, and loss function
            input_dim = dataset.num_features
            hidden_dim = 64
            output_dim = dataset.num_classes

            self.model = GCN(input_dim, hidden_dim, output_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.NLLLoss()
        else:
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.train_loader = train_loader
            self.test_loader = test_loader

    # Training loop
    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to('cuda')  # Move data to GPU
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    # Test function
    def eval(self, loader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data in loader:
                data = data.to('cuda')
                out = self.model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
        return correct / len(loader.dataset)

    def train_eval(self, epochs):
        # Train and evaluate the model
        self.model = self.model.to('cuda')  # Move model to GPU
        for epoch in range(1, epochs+1):
            loss = self.train()
            train_acc = self.eval(self.train_loader)
            test_acc = self.eval(self.test_loader)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    def save_model(self, path='model/victim_model.pth'):
        torch.save(self.model.state_dict(), path)


if __name__ == '__main__':
    # Train and evaluate the model
    trainer = GCNTrainer()
    trainer.train_eval(20)
    trainer.save_model()