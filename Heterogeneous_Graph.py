import os.path as osp
import tqdm
import torch.nn.functional as F

import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from torch_geometric.nn.kge import TransE, DistMult, ComplEx
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.kge.loader import KGTripletLoader
from torch_geometric.loader import LinkNeighborLoader
from create_my_data import create_dataset

# create dataset


data = create_dataset()

# add reverse edges, so the the model is able to pass messages in both directions
data = T.ToUndirected()(data)

# Split the data into train, validation and test set
# TODO: better splitting algorithm
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("module", "directlyConnectedTo", "module"),
    rev_edge_types=("module", "directlyConnectedTo", "module"), 
)
train_data, val_data, test_data = transform(data)

# TODO: different loader for one tree (substation -> bay -> [module1, module2, ...])
# Load the data so that every sample is a tree with the following structure: (substation -> bay -> [module1, module2, ...])


train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10, 5], # 20 neighbors in the first hop, 10 in the second, 5 in the third
    neg_sampling_ratio=2.0,
    edge_label_index=(("module", "directlyConnectedTo", "module"), train_data['directlyConnectedTo'].edge_label_index),
    edge_label=train_data['directlyConnectedTo'].edge_label,
    batch_size=128,
    shuffle=True,
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10, 5], # 20 neighbors in the first hop, 10 in the second, 5 in the third
    neg_sampling_ratio=2.0,
    edge_label_index=(("module", "directlyConnectedTo", "module"), val_data['directlyConnectedTo'].edge_label_index),
    edge_label=val_data['directlyConnectedTo'].edge_label,
    batch_size=3*128,
    shuffle=False,
)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
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
    # TODO: learn maths

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin_sub = torch.nn.Linear(data['substation'].x.size(1), hidden_channels) # linear layer to transform the node features (num_feats, hidden_channels) -> output hidden_channels_dim
        self.emb_sub = torch.nn.Embedding(data["substation"].num_nodes, hidden_channels) # embedding layer to transform the node ids (num_nodes, hidden_channels)

        self.lin_bay = torch.nn.Linear(data['bay'].x.size(1), hidden_channels) # linear layer to transform the node features (num_feats, hidden_channels)
        self.emb_bay = torch.nn.Embedding(data["bay"].num_nodes, hidden_channels) # embedding layer to transform the node ids (num_nodes, hidden_channels)

        self.lin_module = torch.nn.Linear(data['module'].x.size(1), hidden_channels) # linear layer to transform the node features (num_feats, hidden_channels)
        self.emb_module = torch.nn.Embedding(data["module"].num_nodes, hidden_channels) # embedding layer to transform the node ids (num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        # Extract node features:
        x_dict = {
            "substation": self.lin_sub(data['substation'].x) + self.emb_sub(data["substation"].node_id), # feature matrix of modules
            "bay": self.lin_bay(data['bay'].x) + self.emb_bay(data["bay"].node_id), # feature matrix of modules
            "module": self.lin_module(data['module'].x) + self.emb_module(data["module"].node_id), # feature matrix of modules
        }
        # `edge_index_dict` holds all edge indices of all edge types
        print(x_dict)
        print(data.edge_index_dict)
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict['module'],
            data["module", "directlyConnectedTo", "module"].edge_label_index,
        )
        return pred
        

# TODO: use test data
def train():
    model.train()
    for epoch in range(1, 50):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            print(sampled_data)
            optimizer.zero_grad()
            pred = model(sampled_data)
            # TODO: test
            ground_truth = sampled_data["directlyConnectedTo"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        loss = total_loss / total_examples
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'model.pt')



@torch.no_grad()
def val():
    model.eval()
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(val_loader):
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["directlyConnectedTo"].edge_label)
    pred = torch.cat(preds, dim=0)
    pred = (pred > 0.5).float().numpy() # thresholding
    ground_truth = torch.cat(ground_truths, dim=0)
    ground_truth = (ground_truth > 0.5).float().numpy() # thresholding 
    auc = accuracy_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'auc': auc,
        }, 'model.pt')


model = Model(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model.load_state_dict(torch.load('model.pt')['model_state_dict'])
train()
val()

