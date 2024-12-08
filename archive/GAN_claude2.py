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


class Generator(nn.Module):
    def __init__(self, latent_dim, num_nodes, node_features, temperature=.5):
        super(Generator, self).__init__()
        
        # Store dimensions
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.temperature = temperature
        
        # MLPs for generating logits
        self.adj_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * num_nodes)
        )
        
        self.feat_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256), 
            nn.ReLU(),
            nn.Linear(256, num_nodes * node_features)
        )

    def gumbel_softmax(self, logits, temperature=1.0, hard=True, dim=-1):
        """
        Sample from Gumbel-Softmax distribution
        """
        # Sample from Gumbel distribution
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        
        # Softmax
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through trick
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
            
        return ret

    def gumbel_sigmoid(self, logits, temperature=1.0, hard=True):
        """
        Sample from Gumbel-Sigmoid distribution
        """
        # Sample from Gumbel distribution
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        y = torch.sigmoid(y / temperature)
        if hard:
            y_hard = (y > 0.5).float()
            y = y_hard - y.detach() + y
        return y

    def forward(self, z):
        batch_size = z.size(0)
        
        # Generate adjacency matrix logits
        adj_logits = self.adj_mlp(z)
        adj_logits = adj_logits.view(-1, self.num_nodes, self.num_nodes)
        
        # Apply Gumbel-Sigmoid to get adjacency matrix entries between 0 and 1
        adj = self.gumbel_sigmoid(adj_logits, self.temperature)
        
        # Make adjacency matrix symmetric
        adj = (adj + adj.transpose(1, 2)) / 2
        adj = (adj > 0.7).float()  # Ensure the adjacency matrix is binary
        adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2)) # Remove self-loops

        # Generate node feature logits 
        feat_logits = self.feat_mlp(z)
        feat_logits = feat_logits.view(-1, self.num_nodes, self.node_features)
        
        # Apply Gumbel-Softmax to get one-hot node features
        features = self.gumbel_softmax(feat_logits, self.temperature, dim=-1)
        
        return adj, features
        
    def sample_graphs(self, num_samples, device='cuda'):
        """Generate multiple graph samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.forward(z)

# Updated Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_nodes, node_features):
        super(Discriminator, self).__init__()
        
        self.num_nodes = num_nodes
        self.node_features = node_features
        
        input_dim = num_nodes * node_features + num_nodes * num_nodes
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, adj):
        # x: (batch_size, num_nodes, node_features)
        # adj: (batch_size, num_nodes, num_nodes)
        
        batch_size = x.size(0)
        
        x_flat = x.view(batch_size, -1)
        adj_flat = adj.view(batch_size, -1)
        
        input = torch.cat([x_flat, adj_flat], dim=1)
        
        output = self.fc(input)
        return output  # Note: Removed sigmoid here for WGAN-GP
    
# Define a simple autoencoder for graph reconstruction
class GraphAutoencoder(nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim=64):
        super(GraphAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features

        # Encoder: transforms adjacency + features into a latent representation
        # We'll flatten adjacency and features and encode them.
        input_dim = num_nodes * num_nodes + num_nodes * node_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder: reconstruct adjacency and features from the latent vector
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, features, adj):
        # features: (batch_size, num_nodes, node_features)
        # adj: (batch_size, num_nodes, num_nodes)
        batch_size = features.size(0)

        # Flatten inputs
        features_flat = features.view(batch_size, -1)
        adj_flat = adj.view(batch_size, -1)
        x = torch.cat([features_flat, adj_flat], dim=1)

        # Encode
        z = self.encoder(x)
        # Decode
        x_recon = self.decoder(z)

        # Split reconstructed output back into features and adjacency
        feat_recon = x_recon[:, :self.num_nodes * self.node_features]
        adj_recon = x_recon[:, self.num_nodes * self.node_features:]

        feat_recon = feat_recon.view(batch_size, self.num_nodes, self.node_features)
        adj_recon = adj_recon.view(batch_size, self.num_nodes, self.num_nodes)

        return feat_recon, adj_recon

    def reconstruction_loss(self, features, adj, feat_recon, adj_recon):
        # For adjacency, we can use a binary cross-entropy or MSE loss.
        # For features, since they're one-hot, we can also use cross-entropy or MSE.
        # Here, we use MSE for simplicity.
        feat_loss = F.mse_loss(feat_recon, features)
        adj_loss = F.mse_loss(adj_recon, adj)
        return feat_loss + adj_loss


# Define the GCN model (victim model)
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

class UnboundAttack:
    def __init__(self, latent_dim, num_nodes, node_features, victim_model, device='cpu', beta=0.5, recon_weight=0.1):
        self.generator = Generator(latent_dim, num_nodes, node_features).to(device)
        self.discriminator = Discriminator(num_nodes, node_features).to(device)
        self.autoencoder = GraphAutoencoder(num_nodes, node_features).to(device)
        self.victim_model = victim_model
        self.beta = beta
        self.device = device
        self.num_nodes = num_nodes
        self.recon_weight = recon_weight  # Weight for reconstruction loss term

        # Freeze victim model weights
        for param in self.victim_model.parameters():
            param.requires_grad = False

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
    
    def pretrain_autoencoder(self, real_loader):
        # Pre-train the autoencoder on real data
        self.autoencoder.train()
        for data in real_loader:
            data = data.to(self.device)
            self.ae_optimizer.zero_grad()
            real_x_padded, _ = to_dense_batch(data.x, data.batch, max_num_nodes=self.num_nodes)
            real_adj_padded = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.num_nodes)
            real_x_padded = real_x_padded.to(self.device)
            real_adj_padded = real_adj_padded.to(self.device)

            feat_recon, adj_recon = self.autoencoder(real_x_padded, real_adj_padded)
            loss = self.autoencoder.reconstruction_loss(real_x_padded, real_adj_padded, feat_recon, adj_recon)
            loss.backward()
            self.ae_optimizer.step()
            
    def train_step(self, batch, target_class):
        batch_size = batch.num_graphs
        n_critic = 1

        for _ in range(n_critic):
            # ---------------------
            # Train Discriminator
            # ---------------------
            self.d_optimizer.zero_grad()

            # Generate fake graphs
            z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
            fake_adj, fake_features = self.generator(z)

            # fake_adj: (batch_size, num_nodes, num_nodes)
            # fake_features: (batch_size, num_nodes, node_features)

            # Prepare real data
            real_x = batch.x.to(self.device)
            real_edge_index = batch.edge_index.to(self.device)
            real_batch = batch.batch.to(self.device)

            # Convert real data to dense format with padding
            real_x_padded, _ = to_dense_batch(real_x, real_batch, max_num_nodes=self.num_nodes)
            real_adj_padded = to_dense_adj(real_edge_index, real_batch, max_num_nodes=self.num_nodes)

            # Ensure real_x_padded and real_adj_padded are on the same device
            real_x_padded = real_x_padded.to(self.device)
            real_adj_padded = real_adj_padded.to(self.device)

            # Get predictions from discriminator
            real_pred = self.discriminator(real_x_padded, real_adj_padded)
            fake_pred = self.discriminator(fake_features, fake_adj)

            # Compute WGAN loss
            d_loss = torch.mean(fake_pred) - torch.mean(real_pred)  # WGAN discriminator loss

            # Gradient penalty
            epsilon = torch.rand(batch_size, 1, 1).to(self.device)
            interpolated_x = epsilon * real_x_padded + (1 - epsilon) * fake_features
            interpolated_adj = epsilon * real_adj_padded + (1 - epsilon) * fake_adj
            interpolated_x.requires_grad_(True)
            interpolated_adj.requires_grad_(True)

            interpolated_pred = self.discriminator(interpolated_x, interpolated_adj)

            grad_outputs = torch.ones_like(interpolated_pred).to(self.device)

            # Compute gradients with respect to inputs
            gradients = torch.autograd.grad(
                outputs=interpolated_pred,
                inputs=[interpolated_x, interpolated_adj],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )

            gradient_x = gradients[0].view(batch_size, -1)
            gradient_adj = gradients[1].view(batch_size, -1)
            gradient_norm = torch.sqrt((gradient_x ** 2).sum(dim=1) + (gradient_adj ** 2).sum(dim=1) + 1e-12)
            gradient_penalty = ((gradient_norm - 1) ** 2).mean()
            lambda_gp = 10  # You can adjust this value

            # Update discriminator loss with gradient penalty
            d_loss = d_loss + lambda_gp * gradient_penalty
            d_loss.backward()
            self.d_optimizer.step()

        # ---------------------
        # Train Generator
        # ---------------------
        self.g_optimizer.zero_grad()

        # Generate new fake graphs
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_adj, fake_features = self.generator(z)

        # Get predictions from discriminator
        fake_pred = self.discriminator(fake_features, fake_adj)

        # Prepare fake data for victim model (convert to sparse format)
        data_list = []
        for i in range(batch_size):
            adj_i = fake_adj[i]  # num_nodes x num_nodes
            features_i = fake_features[i]  # num_nodes x node_features

            # Threshold adjacency matrix to get binary values
            adj_i = (adj_i > 0.5).to(torch.long)

            # Get edge indices from adjacency matrix
            edge_index_i = adj_i.nonzero(as_tuple=False).t().contiguous()

            # Create Data object
            data_i = Data(x=features_i, edge_index=edge_index_i)
            data_list.append(data_i)

        # Create a batch from the list of Data objects
        fake_batch = Batch.from_data_list(data_list).to(self.device)

        # Get predictions from victim model
        victim_pred = self.victim_model(fake_batch.x, fake_batch.edge_index, fake_batch.batch)

        # Adversarial loss
        g_loss = -torch.mean(fake_pred)
        probabilities = torch.exp(victim_pred)
        adv_loss = F.relu(0.5 - probabilities[:, target_class])
        # adv_loss = F.relu(0.5 - victim_pred[:, target_class])

        # Autoencoder reconstruction loss on fake data
        self.autoencoder.eval()  # Assuming AE is pretrained or fixed; set to train if you want joint training
        with torch.no_grad():
            feat_recon, adj_recon = self.autoencoder(fake_features.detach(), fake_adj.detach())
        # Since we used detach(), these won't affect AE's parameters, only G. If you want AE to train as well, remove detach().
        recon_loss = self.autoencoder.reconstruction_loss(fake_features, fake_adj, feat_recon, adj_recon)
        

        total_g_loss = g_loss + self.beta * adv_loss.mean() + self.recon_weight * recon_loss
        total_g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), total_g_loss.item()

    def generate_attack(self, num_samples, target_class):
        """Generate adversarial graph examples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.generator.latent_dim).to(self.device)
            fake_adj, fake_features = self.generator(z)
        return fake_adj, fake_features

# Load the MUTAG dataset
dataset = TUDataset(root='data/MUTAG', name='MUTAG')

# Shuffle and split the dataset into training and test sets
torch.manual_seed(42)
dataset = dataset.shuffle()
train_size = int(len(dataset) * 0.8)
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, optimizer, and loss function
input_dim = dataset.num_features
hidden_dim = 64
output_dim = dataset.num_classes

model = GCN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to('cuda')  # Move data to GPU
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Test function
def test(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to('cuda')
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# Train and evaluate the model
model = model.to('cuda')  # Move model to GPU
for epoch in range(1, 30):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Example usage:
# Initialize attack framework and hyperparameters
attack = UnboundAttack(
    latent_dim=100,
    num_nodes=28,  # Set to maximum number of nodes in the dataset
    node_features=input_dim,  # dataset.num_features ensures consistency
    victim_model=model,
    device='cuda',
    beta=0.1,
    recon_weight=2.0
)
print("Pretraining autoencoder...")
attack.pretrain_autoencoder(train_loader)

# Training loop
for epoch in range(10):
    for batch in train_loader:
        batch = batch.to('cuda')
        d_loss, g_loss = attack.train_step(batch, target_class=0)
        print(f'Epoch {epoch}: D_loss {d_loss}, G_loss {g_loss}')

# Generate adversarial examples
fake_adj, fake_features = attack.generate_attack(num_samples=10, target_class=0)
print("Fake Examples:")
print("Adjacency matrices:", fake_adj)
print("Node features:", fake_features)

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
    else:
        first_matrix = first_matrix.cpu().numpy()
    # Create a graph from the adjacency matrix
    graph = nx.from_numpy_array(first_matrix)
    
    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(graph, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold", edge_color="gray")
    plt.title("Graph Visualization of the First Adjacency Matrix")
    plt.savefig(f_name, format="png", dpi=300)
    plt.close()

# Call the function with t_adj
visualize_adjacency_matrix(fake_adj, 'graph_vis_fake.png')
visualize_adjacency_matrix(real_adj, 'graph_vis_real.png')