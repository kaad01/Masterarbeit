from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch.utils.data import random_split

class TreeLoader(Dataset):
    def __init__(self, data: HeteroData, **kwargs):
        super().__init__()
        self.trees = self.processed_trees(data)

    def processed_trees(self, data: HeteroData):
        trees = []
        for bay_id in data['bay'].node_id.tolist():
            # bay_data = data['bay'].x[bay_id].view(1, -1)

            connected_substation = data['bay', 'rev_hasPart', 'substation'].edge_index[1]\
                                    [data['bay', 'rev_hasPart', 'substation'].edge_index[0] == bay_id] # get the substation connected to the bay
            substation_data = data['substation'].x[connected_substation].view(1, -1)
            
            modules_mask = data['bay', 'hasPart', 'module'].edge_index[0] == bay_id
            connected_modules = data['bay', 'hasPart', 'module'].edge_index[1][modules_mask] # get the modules connected to the bay
            
            if len(connected_modules) == 0:
                continue
            module_data = data['module'].x[connected_modules]

            module_module_edge_index = data['module', 'directlyConnectedTo', 'module'].edge_index
            relevant_module_edges = torch.logical_and(
                torch.isin(module_module_edge_index[0], connected_modules),
                torch.isin(module_module_edge_index[1], connected_modules)
            )

            remapped_module_edges = module_module_edge_index[:, relevant_module_edges]
            remapped_module_edges[0] = torch.tensor([connected_modules.tolist().index(edge) for edge in remapped_module_edges[0]])
            remapped_module_edges[1] = torch.tensor([connected_modules.tolist().index(edge) for edge in remapped_module_edges[1]])


            tree_data = HeteroData()
            # add nodes
            tree_data['substation'].node_id = torch.tensor([0])
            tree_data['bay'].node_id = torch.tensor([0])
            tree_data['module'].node_id = torch.arange(len(connected_modules))

            tree_data['substation'].x = substation_data
            # tree_data['bay'].x = bay_data
            tree_data['module'].x = module_data

            # add edges
            tree_data['substation', 'hasPart', 'bay'].edge_index = torch.tensor([[0], [0]])
            tree_data['bay', 'hasPart', 'module'].edge_index = torch.tensor([[0]*len(connected_modules), list(range(len(connected_modules)))])
            tree_data['module', 'directlyConnectedTo', 'module'].edge_index = remapped_module_edges

            tree_data = T.ToUndirected()(tree_data)
            
            trees.append(tree_data)
        
        return trees

    def split(self):
        total_size = self.__len__()
        train_size = int(0.7 * total_size)
        valid_size = total_size - train_size
        # valid_size = int(0.2 * total_size)
        # test_size = total_size - train_size - valid_size

        train_dataset, valid_dataset = random_split(self.trees,[train_size, valid_size])
        # train_dataset, valid_dataset, test_dataset = random_split(self.trees,[train_size, valid_size, test_size])

        return train_dataset, valid_dataset

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)

