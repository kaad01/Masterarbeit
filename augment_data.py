import random
import torch
from torch_geometric.data import HeteroData
from torch import cat as concat

def augment_data(data, num_copies=5, max_changes=2):
    """
    Augments the given dataset by creating multiple copies of it, each with random small changes to the node attributes.

    Parameters:
        data (HeteroData): The original dataset to be augmented.
        num_copies (int): The number of copies to create.
        max_changes (int): The maximum number of attribute changes to make in each copy.

    Returns:
        augmented_data (list of HeteroData): The list of augmented datasets.
    """

    new_data = HeteroData()

    for i in range(num_copies):
        node_types = data.node_types

        for node_type in node_types:
            num_nodes = data[node_type].x.shape[0]
            if i == 0:
                new_data[node_type].node_id = torch.arange(num_nodes * num_copies)
                new_data[node_type].x = data[node_type].x.clone()
            else:
                new_data[node_type].x = concat([new_data[node_type].x, data[node_type].x], dim=0)

            # make random changes to the node attributes
            # for j in range(num_nodes):
            #     for k in range(max_changes):
            #         attribute_index = random.randint(0, data[node_type].x.shape[1] - 1)
            #         new_data[node_type].x[j, attribute_index] = torch.randn(1) * 0.1 + data[node_type].x[j, attribute_index]

        # create new edge indices
        substation_bay_edge_index = data['substation', 'hasPart', 'bay'].edge_index.clone()
        substation_bay_edge_index[0] += i * data['substation'].x.shape[0]
        substation_bay_edge_index[1] += i * data['bay'].x.shape[0]

        bay_module_edge_index = data['bay', 'hasPart', 'module'].edge_index.clone()
        bay_module_edge_index[0] += i * data['bay'].x.shape[0]
        bay_module_edge_index[1] += i * data['module'].x.shape[0]

        module_module_edge_index = data['module', 'directlyConnectedTo', 'module'].edge_index.clone()
        module_module_edge_index[0] += i * data['module'].x.shape[0]
        module_module_edge_index[1] += i * data['module'].x.shape[0]

        # randomly remove some edges between modules
        module_module_edge_index_removed = []
        for j in range(module_module_edge_index.shape[1]):
            if random.random() > 0.3: # remove with 30% probability
                module_module_edge_index_removed.append(j)

        module_module_edge_index = module_module_edge_index[:, [j for j in range(module_module_edge_index.shape[1]) if j not in module_module_edge_index_removed]]


         # add edges to new data object
        if i == 0:
            new_data['substation', 'hasPart', 'bay'].edge_index = substation_bay_edge_index
            new_data['bay', 'hasPart', 'module'].edge_index = bay_module_edge_index
            new_data['module', 'directlyConnectedTo', 'module'].edge_index = module_module_edge_index
        else:
            new_data['substation', 'hasPart', 'bay'].edge_index = concat([new_data['substation', 'hasPart', 'bay'].edge_index, substation_bay_edge_index], dim=1)
            new_data['bay', 'hasPart', 'module'].edge_index = concat([new_data['bay', 'hasPart', 'module'].edge_index, bay_module_edge_index], dim=1)
            new_data['module', 'directlyConnectedTo', 'module'].edge_index = concat([new_data['module', 'directlyConnectedTo', 'module'].edge_index, module_module_edge_index], dim=1)


    return new_data

