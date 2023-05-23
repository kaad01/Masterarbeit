from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import torch
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from TreeLoader import TreeLoader
from torch_geometric.utils import negative_sampling, to_networkx
import networkx as nx
from create_my_data import create_dataset
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, RocCurveDisplay
import random
import math

def save(object, name):
    with open(name, 'wb') as f:
        pickle.dump(object, f)
def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# create dataset
#data = create_dataset()

# add reverse edges, so the the model is able to pass messages in both directions
# data = T.ToUndirected()(data)
# save(data, "augmented_data.pkl")
data = load("data_simple.pkl")

# tree_loader = TreeLoader(data)
# save(tree_loader, "augmented_tree_loader.pkl")
tree_loader = load("tree_loader_simple.pkl")

print("TreeLoader loaded")
train_dataset, valid_dataset = tree_loader.split()
train_dataset = list(train_dataset)
valid_dataset = list(valid_dataset)
print(type(train_dataset))

random.shuffle(train_dataset)
random.shuffle(valid_dataset)


# def hetero_collate(data):
#     return Batch.from_data_list(data)
    

# # Erstellen von DataLoadern für jeden Datensatz
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=hetero_collate)
# valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, collate_fn=hetero_collate)




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
        new_tree = tree.clone()
        del new_tree['module', 'directlyConnectedTo', 'module'].edge_index # delete edges
        new_tree['module', 'directlyConnectedTo', 'module'].edge_index = torch.tensor([[],[]], dtype=torch.long) # placeholder for edges
        x = self.gnn(x_dict, new_tree.edge_index_dict)

        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        try:
            pos_pred = self.classifier(z['module'], pos_edge_index)
        except:
            pos_pred = torch.tensor([0])
        try:
            neg_pred = self.classifier(z['module'], neg_edge_index)
        except:
            neg_pred = torch.tensor([0])
        return pos_pred, neg_pred



def generate_positive_negative_examples(edge_index, num_nodes):
    # positive examples
    pos_edge_index = edge_index
    # negative examples
    neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes)

    return pos_edge_index, neg_edge_index



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(hidden_channels=64).to(device)
# model.load_state_dict(torch.load('models/model_real_good.pt')['model_state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

def train():
    model.train()

    total_loss = 0
    skipped = 0
    for data in tqdm(train_dataset):
        data = data.to(device)
        optimizer.zero_grad()
        z = model(data)

        pos_edge_index, neg_edge_index = generate_positive_negative_examples(data['directlyConnectedTo'].edge_index, data['module'].num_nodes)
        pos_preds, neg_preds = model.decode(z, pos_edge_index, neg_edge_index)

        pos_labels = torch.ones(pos_preds.shape[0]).to(device)
        neg_labels = torch.zeros(neg_preds.shape[0]).to(device)

        try:
            loss = F.binary_cross_entropy_with_logits(pos_preds, pos_labels) + F.binary_cross_entropy_with_logits(neg_preds, neg_labels)
        except:
            skipped += 1
            continue

        if not math.isnan(loss.item()):
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            skipped += 1

    return total_loss / (len(train_dataset)-skipped)

def validate():
    model.eval()
    preds = []
    ground_truths = []
    skipped = 0
    with torch.no_grad():
        total_loss = 0
        for data in valid_dataset:
            data = data.to(device)
            z = model(data)
            pos_edge_index, neg_edge_index = generate_positive_negative_examples(data['directlyConnectedTo'].edge_index, data['module'].num_nodes)
            pos_preds, neg_preds = model.decode(z, pos_edge_index, neg_edge_index)
            preds.append(torch.cat([pos_preds, neg_preds], dim=0))

            pos_labels = torch.ones(pos_preds.shape[0]).to(device)
            neg_labels = torch.zeros(neg_preds.shape[0]).to(device)
            ground_truths.append(torch.cat([pos_labels, neg_labels], dim=0))

            try:
                loss = F.binary_cross_entropy_with_logits(pos_preds, pos_labels) + F.binary_cross_entropy_with_logits(neg_preds, neg_labels)
            except:
                skipped += 1
                continue
            if not math.isnan(loss.item()):
                total_loss += loss.item()
            else:
                skipped += 1

    pred = torch.cat(preds, dim=0)
    ground_truth = torch.cat(ground_truths, dim=0)
    auc = roc_auc_score(ground_truth, pred)
    # RocCurveDisplay.from_predictions(ground_truth, pred)

    # plt.show()
    
    pred = (pred > 0.5).float().numpy() # thresholding
    acc = accuracy_score(ground_truth, pred)
    # TODO: für jeden data punkt (Feld)

    loss = total_loss / (len(valid_dataset)- skipped)

    return loss, auc, acc


num_epochs = 10
train_losses = []
validation_accuracies = []
validation_auc = []
for epoch in range(num_epochs):
    val_loss, auc, acc = validate()
    validation_auc.append(auc)
    validation_accuracies.append(acc)

    train_loss = train()
    train_losses.append(train_loss)

    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {acc}, AUC: {auc}")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'validation_accuracies': validation_accuracies,
    'validation_auc': validation_auc
}, 'models/model_real_good_2.pt')


# visualize training performance
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), validation_accuracies, label='Validation Accuracy')
plt.plot(range(1, num_epochs+1), validation_auc, label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Performance')
plt.legend()
plt.show()

