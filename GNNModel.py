# Install required packages for graph neural networks
import subprocess
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    print("PyTorch Geometric already installed")
except ImportError:
    print("Installing PyTorch and PyTorch Geometric...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    print("Successfully installed PyTorch Geometric")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import scipy.sparse as sp

print("Building Graph Neural Network models...")

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create edge index from adjacency matrix
adj_coo = adj_matrix.tocoo()
edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)

# Create PyTorch Geometric data object
x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor(node_labels, dtype=torch.long)

# Create masks for applications only (since we're predicting fraud on applications)
application_mask = []
for i, node in enumerate(node_list):
    if G.nodes[node]['node_type'] == 'application':
        application_mask.append(i)
    else:
        application_mask.append(-1)

application_indices = [i for i in application_mask if i != -1]
application_indices = torch.tensor(application_indices, dtype=torch.long)

# Split data for training/validation/test
train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
n_applications = len(application_indices)
n_train = int(train_ratio * n_applications)
n_val = int(val_ratio * n_applications)

# Shuffle application indices
perm = torch.randperm(n_applications)
train_idx = application_indices[perm[:n_train]]
val_idx = application_indices[perm[n_train:n_train+n_val]]
test_idx = application_indices[perm[n_train+n_val:]]

print(f"Training applications: {len(train_idx)}")
print(f"Validation applications: {len(val_idx)}")
print(f"Test applications: {len(test_idx)}")

# Check class distribution
train_labels = y[train_idx]
print(f"Training set fraud rate: {train_labels.float().mean():.3f}")

# Create data object
data = Data(x=x, edge_index=edge_index, y=y)
data = data.to(device)

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.5):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim//heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize models
input_dim = x.shape[1]
hidden_dim = 64
output_dim = 2  # Binary classification

models = {
    'GCN': GCNModel(input_dim, hidden_dim, output_dim),
    'GraphSAGE': GraphSAGEModel(input_dim, hidden_dim, output_dim),
    'GAT': GATModel(input_dim, hidden_dim, output_dim)
}

print("Models initialized successfully")
print(f"Input dimension: {input_dim}")
print(f"Hidden dimension: {hidden_dim}")
print(f"Output dimension: {output_dim}")