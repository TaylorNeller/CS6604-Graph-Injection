import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj

# Load the MUTAG dataset
dataset = TUDataset(root='data/MUTAG', name='MUTAG')
loader = DataLoader(dataset, batch_size=1, shuffle=True)  # Set batch_size=1

# Define the victim GCN classifier
class GCNClassifier(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Initialize and train the victim classifier
def train_classifier():
    classifier = GCNClassifier(num_features=dataset.num_node_features,
                               hidden_dim=64,
                               num_classes=dataset.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)

    classifier.train()
    for epoch in range(20):
        for data in loader:
            optimizer.zero_grad()
            out = classifier(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
    return classifier

victim_classifier = train_classifier()

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, noise_dim, num_features):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_features = num_features

        # MLP to generate node features
        self.node_mlp = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_features)
        )

        # MLP to generate edge probabilities
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z, num_nodes):
        # Generate node features
        batch_size = z.size(0)
        node_z = torch.randn(batch_size, num_nodes, self.noise_dim).to(z.device)
        node_features = self.node_mlp(node_z)  # [batch_size, num_nodes, num_features]

        # Generate adjacency matrix
        node_features_expanded = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_features_transposed = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        edge_input = torch.cat([node_features_expanded, node_features_transposed], dim=-1)
        edge_probs = torch.sigmoid(self.edge_mlp(edge_input)).squeeze(-1)
        adj = (edge_probs + edge_probs.transpose(1, 2)) / 2  # Symmetrize

        return node_features, adj

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out

# Wasserstein GAN with Gradient Penalty loss functions
def gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.x.size(0), 1).to(device)
    interpolates_x = (alpha * real_data.x + ((1 - alpha) * fake_data.x)).requires_grad_(True)
    interpolates_edge_index = real_data.edge_index  # Keeping edge_index fixed

    interpolates_batch = real_data.batch

    d_interpolates = discriminator(interpolates_x, interpolates_edge_index, interpolates_batch)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates_x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(noise_dim=128, num_features=dataset.num_node_features).to(device)
discriminator = Discriminator(num_features=dataset.num_node_features, hidden_dim=64).to(device)
victim_classifier.to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
lambda_gp = 10  # Gradient penalty lambda

for epoch in range(100):
    for i, real_data in enumerate(loader):
        real_data = real_data.to(device)
        num_nodes = real_data.num_nodes  # Number of nodes in the graph

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real data
        d_real = discriminator(real_data.x, real_data.edge_index, real_data.batch)
        d_real_loss = -torch.mean(d_real)

        # Fake data
        z = torch.randn(1, 128).to(device)  # batch_size=1
        fake_x, fake_adj = generator(z, num_nodes)

        # Convert adjacency matrix to edge index
        adj = fake_adj.squeeze(0)  # [num_nodes, num_nodes]
        edge_index = (adj > 0.5).nonzero(as_tuple=False).t().contiguous()
        edge_index = edge_index.to(device)

        fake_batch = torch.zeros(fake_x.size(1), dtype=torch.long).to(device)  # batch_size=1
        fake_data = Data(x=fake_x.squeeze(0), edge_index=edge_index, batch=fake_batch)

        d_fake = discriminator(fake_data.x, fake_data.edge_index, fake_data.batch)
        d_fake_loss = torch.mean(d_fake)

        # Gradient penalty
        gp = gradient_penalty(discriminator, real_data, fake_data)

        d_loss = d_real_loss + d_fake_loss + lambda_gp * gp
        d_loss.backward()
        optimizer_D.step()

        # Train Generator every n_critic steps
        if i % 5 == 0:
            optimizer_G.zero_grad()

            z = torch.randn(1, 128).to(device)
            fake_x, fake_adj = generator(z, num_nodes)

            adj = fake_adj.squeeze(0)
            edge_index = (adj > 0.5).nonzero(as_tuple=False).t().contiguous()
            edge_index = edge_index.to(device)

            fake_batch = torch.zeros(fake_x.size(1), dtype=torch.long).to(device)
            fake_data = Data(x=fake_x.squeeze(0), edge_index=edge_index, batch=fake_batch)

            # Adversarial loss
            g_adv = -torch.mean(discriminator(fake_data.x, fake_data.edge_index, fake_data.batch))

            # Misclassification loss
            victim_classifier.eval()
            with torch.no_grad():
                victim_outputs = victim_classifier(fake_data.x, fake_data.edge_index, fake_data.batch)
                target_labels = torch.randint(0, dataset.num_classes, (victim_outputs.size(0),)).to(device)
            g_misclassify = F.cross_entropy(victim_outputs, target_labels)

            g_loss = g_adv + g_misclassify
            g_loss.backward()
            optimizer_G.step()

    print(f'Epoch {epoch}: D_loss {d_loss.item()}, G_loss {g_loss.item()}')

