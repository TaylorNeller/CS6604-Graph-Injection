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
    def __init__(self, latent_dim, num_nodes, node_features, temperature=.8):
        super(Generator, self).__init__()
        
        # Store dimensions
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.temperature = temperature
        self.hard = False

        self.adj_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes * num_nodes)
        )
        
        self.feat_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes * node_features)
        )
    
    def set_hard(self, hard):
        self.hard = hard

    def gumbel_softmax(self, logits, temperature=1.0, dim=-1):
        """
        Sample from Gumbel-Softmax distribution
        """
        # Sample from Gumbel distribution
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        
        # Softmax
        y_soft = gumbels.softmax(dim)

        if self.hard:
            # Straight through trick
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
            
        return ret

    def gumbel_sigmoid(self, logits, temperature=1.0):
        """
        Sample from Gumbel-Sigmoid distribution
        """
        # Sample from Gumbel distribution
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        y = torch.sigmoid(y / temperature)
        if self.hard:
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
        # adj = (adj > 0.5).float()  # Ensure the adjacency matrix is binary
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

class UnboundAttack:
    def __init__(self, latent_dim, num_nodes, node_features, victim_model, target_degree_dist, device='cpu', beta=0.5, lambda_degree=10.0, n_gen=1, n_critic=1, lambda_=1, temperature=.7):
        self.generator = Generator(latent_dim, num_nodes, node_features, temperature=temperature).to(device)
        self.discriminator = Discriminator(num_nodes, node_features).to(device)
        self.victim_model = victim_model.to(device)
        self.beta = beta
        self.device = device
        self.num_nodes = num_nodes
        self.lambda_degree = lambda_degree
        self.target_degree_dist = target_degree_dist
        self.n_gen = n_gen
        self.n_critic = n_critic
        self.lambda_ = lambda_

        # Freeze victim model weights
        for param in self.victim_model.parameters():
            param.requires_grad = False

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    
    def differentiable_degree_loss(self, adj_batch, sigma=0.5):
        """
        Compute a differentiable loss that encourages the generated graphs to have 
        a degree distribution that matches a given target distribution.

        Parameters:
            adj_batch (torch.Tensor): Continuous adjacency matrices of shape (batch_size, num_nodes, num_nodes).
                                    Values should be in [0,1], representing the probability of an edge.
            target_degree_dist (torch.Tensor): Target degree distribution of shape (max_degree+1).
                                            Example: torch.tensor([0.0, 0.1946, 0.4034, 0.4017, 0.0002966, 0, 0, ...], device=adj.device)
            sigma (float): Standard deviation for the Gaussian kernel used for soft binning.

        Returns:
            torch.Tensor: A scalar loss value encouraging the degree distribution to match the target.
        """
        batch_size, num_nodes, _ = adj_batch.shape

        # Compute continuous degrees for each node
        # degrees: shape (batch_size, num_nodes)
        degrees = adj_batch.sum(dim=-1)

        # Determine maximum degree from the target distribution
        max_degree = self.target_degree_dist.size(0) - 1
        bin_centers = torch.arange(0, max_degree + 1, device=adj_batch.device).float()

        # degrees: (batch_size, num_nodes)
        # We'll convert each degree into a probability distribution over bins using a Gaussian kernel
        # shape after expansion: degrees: (batch_size, num_nodes, 1), bin_centers: (1, 1, max_degree+1)
        deg_expanded = degrees.unsqueeze(-1)  # (batch_size, num_nodes, 1)
        bin_centers_expanded = bin_centers.view(1, 1, -1)  # (1, 1, max_degree+1)

        # Compute squared difference (batch_size, num_nodes, max_degree+1)
        diff_squared = (deg_expanded - bin_centers_expanded) ** 2
        # Apply Gaussian kernel
        weights = torch.exp(-diff_squared / (2 * sigma * sigma))

        # Normalize to get a proper probability distribution over bins per node
        # sum over bins: keepdim=True to maintain shape
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

        # Now we have a distribution over degrees per node. We want the mean distribution over all nodes and all graphs.
        # Average over nodes and batch. If your batch_size = B and num_nodes = N:
        # weights: (B, N, max_degree+1)
        # First average over nodes and batch
        # After averaging: (max_degree+1)
        empirical_degree_dist = weights.mean(dim=1).mean(dim=0)

        # Compare empirical distribution to target distribution
        # MSE is a simple choice, you could also try KL-divergence
        degree_loss = F.mse_loss(empirical_degree_dist, self.target_degree_dist)
        return degree_loss

        # # Add epsilon to avoid log(0) in KL divergence
        # empirical_smoothed = empirical_degree_dist + epsilon
        # target_smoothed = self.target_degree_dist + epsilon

        # # Compute KL divergence
        # # kl_div = (empirical_smoothed * torch.log(empirical_smoothed / target_smoothed)).sum()

        # # Compute weights for rare bins
        # rare_bin_weights = torch.where(self.target_degree_dist < 1e-3, 10.0, 1.0) 
        # kl_div = (rare_bin_weights * empirical_smoothed * torch.log(empirical_smoothed / target_smoothed)).sum()
        # return kl_div

    def train_step(self, batch, target_class):
        batch_size = batch.num_graphs

        for _ in range(self.n_critic):
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

        for _ in range(self.n_gen):
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

            # Distribution matching loss
            degree_loss = self.differentiable_degree_loss(fake_adj)

            # Adversarial loss
            g_loss = -torch.mean(fake_pred)
            probabilities = torch.exp(victim_pred)
            adv_loss = F.relu(0.5 - probabilities[:, target_class])
            # adv_loss = F.relu(0.5 - victim_pred[:, target_class])

            total_g_loss = self.lambda_*(g_loss + self.beta * adv_loss.mean() + self.lambda_degree * degree_loss)
            total_g_loss.backward()
            self.g_optimizer.step()

        return d_loss.item(), total_g_loss.item()

    def generate_attack(self, num_samples):
        """Generate adversarial graph examples"""
        self.generator.eval()
        self.generator.set_hard(True)
        with torch.no_grad():
            z = torch.randn(num_samples, self.generator.latent_dim).to(self.device)
            fake_adj, fake_features = self.generator(z)
        self.generator.set_hard(False)
        return fake_adj, fake_features

    def save_models(self, dir):
        # Save models after training
        torch.save(self.generator.state_dict(), f'{dir}/generator.pth')
        torch.save(self.discriminator.state_dict(), f'{dir}/discriminator.pth')
        torch.save(self.victim_model.state_dict(), f'{dir}/victim_model_backup.pth')
        torch.save(self.g_optimizer.state_dict(), f'{dir}/g_optimizer.pth')
        torch.save(self.d_optimizer.state_dict(), f'{dir}/d_optimizer.pth')

    def load_models(self, dir):
        # Load models for inference
        self.generator.load_state_dict(torch.load(f'{dir}/generator.pth'))
        self.discriminator.load_state_dict(torch.load(f'{dir}/discriminator.pth'))
        self.victim_model.load_state_dict(torch.load(f'{dir}/victim_model.pth'))
        self.g_optimizer.load_state_dict(torch.load(f'{dir}/g_optimizer.pth'))
        self.d_optimizer.load_state_dict(torch.load(f'{dir}/d_optimizer.pth'))

        # to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)  
        self.victim_model.to(self.device)

