from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import torch
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from TreeLoader import TreeLoader
from torch_geometric.utils import negative_sampling
from create_my_data import create_dataset
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import random

def save(object, name):
    with open(name, 'wb') as f:
        pickle.dump(object, f)
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

data = load("data_simple.pkl")
tree_loader = load("tree_loader_simple.pkl")
trees = tree_loader.trees
random.shuffle(trees) # shuffle

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, add_self_loops=False)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_module: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_module1 = x_module[edge_label_index[0]] # module1
        edge_feat_module2 = x_module[edge_label_index[1]] # module2
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_module1 * edge_feat_module2).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin_sub = torch.nn.Linear(data['substation'].x.size(1), hidden_channels) # linear layer to transform the node features (num_feats, hidden_channels) -> output hidden_channels_dim
        self.emb_sub = torch.nn.Embedding(data["substation"].num_nodes, hidden_channels) # embedding layer to transform the node ids (num_nodes, hidden_channels)

        self.emb_bay = torch.nn.Embedding(data["bay"].num_nodes, hidden_channels) # embedding layer to transform the node ids (num_nodes, hidden_channels)

        self.lin_module = torch.nn.Linear(data['module'].x.size(1), hidden_channels) # linear layer to transform the node features (num_feats, hidden_channels)
        self.emb_module = torch.nn.Embedding(data["module"].num_nodes, hidden_channels) # embedding layer to transform the node ids (num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        
        self.classifier = Classifier()


    def forward(self, tree) -> Tensor:
        # Extract node features:
        x_dict = {
            "substation": self.lin_sub(tree['substation'].x) + self.emb_sub(tree["substation"].node_id),
            "bay": self.emb_bay(tree["bay"].node_id),
            "module": self.lin_module(tree['module'].x) + self.emb_module(tree["module"].node_id)
        }
        # `edge_index_dict` holds all edge indices of all edge types
        x = self.gnn(x_dict, tree.edge_index_dict)

        return x

    def decode(self, z, edges):
        pred = self.classifier(z['module'], edges)
        return pred

def generate_edges(tree):
    module_indices = tree['module'].node_id.tolist()
    # from every module to every other module
    edge_0 = [[module]*(len(module_indices)-1) for module in module_indices]
    edge_1 = [module_indices[:i] + module_indices[i+1:] for i in range(len(module_indices))]
    # create edge_index tensor
    edge_index = torch.stack((torch.tensor(edge_0).flatten(), torch.tensor(edge_1).flatten()))
    return edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(hidden_channels=64).to(device)
model.load_state_dict(torch.load('models/model_good.pt')['model_state_dict'])

for tree in trees[:3]:
    print('\nThis is a tree')
    print(tree['module', 'directlyConnectedTo', 'module'].edge_index)
    # del tree['module', 'directlyConnectedTo', 'module'].edge_index # delete edges
    # tree['module', 'directlyConnectedTo', 'module'].edge_index = torch.tensor([[],[]], dtype=torch.long) # placeholder for edges

    tree = tree.to(device)
    z = model(tree)
    edges = generate_edges(tree) # generate all possible edges
    print(edges)
    raw_preds = model.decode(z, edges)
    print(raw_preds)
    preds = (raw_preds > 0.5).float().numpy() # thresholding
    print(preds)

    
