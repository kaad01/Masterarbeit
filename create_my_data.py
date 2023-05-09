import torch
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def create_dataset():
    node_paths = ['data/tables/primtech_Gisswitchgear_instances_properties.csv', 'data/tables/primtech_Bay_instances_properties.csv', 
            'data/tables/primtech_Gismodule_instances_properties_unique.csv']
    edge_paths = ['data/tables/primtech_switchgear_bay_relation.csv', 'data/tables/primtech_bay_module_relation.csv',
                'data/tables/primtech_module_module_relation.csv']


    def load_node_csv(path, index_col, **kwargs):
        df = pd.read_csv(path, index_col=index_col, **kwargs) # read the data
        mapping = {index: i for i, index in enumerate(df.index.unique())} # create a mapping for the data
        encoders = {}
        x = []

        # choose the attributes to use based on node type
        match index_col:
            case'Gisswitchgear_iri':
                df = df[['switchgear-asset-identifier-code','rated-voltage-ur','country-of-installation','end-customer',
                         'switchgear-installation-location-type','switchgear-product-type']]
            case 'Bay_iri':
                df = df
            case 'Gismodule_iri':
                df = df[['customer-asset-identifier','standard-asset-identifier','factory','gis-module-8-digit-code',
                         'primtech-module-matchcode','gis-module-current-phase','gis-module-product-affiliation']]

        # create an encoder for every attribute
        # TODO: different encoders for different attributes
        for col in df.columns:
            encoder = OneHotEncoder(handle_unknown='ignore')
            xs = encoder.fit_transform(df[col].values.reshape(-1, 1)).toarray() # encode the data
            x.append(torch.from_numpy(xs).to(torch.float))
            # encoders.append(encoder)

        # embeddings = {}
        # for col in df.columns:
        #     unique_values = df[col].unique()
        #     label_encoder = LabelEncoder()
        #     label_encoder.fit(unique_values)
        #     encoders[col] = label_encoder

        #     num_embeddings = len(unique_values)
        #     encoder = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=64)
        #     embeddings[col] = encoder

        # # Kodieren Sie die Daten
        # x = []
        # for _, row in df.iterrows():
        #     row_embedding = []
        #     for col in df.columns:
        #         category_idx = encoders[col].transform([row[col]])[0]
        #         category_idx_tensor = torch.tensor([category_idx], dtype=torch.long)
        #         category_embedding = embeddings[col](category_idx_tensor)
        #         row_embedding.append(category_embedding.squeeze())

        #     row_embedding_tensor = torch.cat(row_embedding, dim=-1)
        #     x.append(row_embedding_tensor)

        x = torch.cat(x, dim=-1)
        return x, mapping

    def load_edge_csv(path, source_index_col, source_mapping, target_index_col, target_mapping, **kwargs):
        df = pd.read_csv(path, **kwargs)
        source = [source_mapping[index] for index in df[source_index_col]] # get the source indices
        target = [target_mapping[index] for index in df[target_index_col]] # get the target indices
        edge_index = torch.tensor([source, target], dtype=torch.long) # create the edge_index
        return edge_index

    # get node data
    substation_x, substation_mapping = load_node_csv(node_paths[0], 'Gisswitchgear_iri', dtype=str)
    bay_x, bay_mapping = load_node_csv(node_paths[1], 'Bay_iri', dtype=str)
    module_x, module_mapping = load_node_csv(node_paths[2], 'Gismodule_iri', dtype=str)

    # get edge data
    substation_bay_edge_index = load_edge_csv(edge_paths[0], 'substation_iri', substation_mapping, 'bay_iri', bay_mapping, dtype=str)
    bay_module_edge_index = load_edge_csv(edge_paths[1], 'bay_iri', bay_mapping, 'gis-module_iri', module_mapping, dtype=str)
    module_module_edge_index = load_edge_csv(edge_paths[2], 'gis-module_iri', module_mapping, 'gis-module-2_iri', module_mapping, dtype=str)

    # create a HeteroData object
    data = HeteroData()

    # add nodes
    data['substation'].node_id = torch.arange(len(substation_mapping))
    data['bay'].node_id = torch.arange(len(bay_mapping))
    data['module'].node_id = torch.arange(len(module_mapping))

    data['substation'].x = substation_x
    data['bay'].x = bay_x
    data['module'].x = module_x

    # add edges
    data['substation', 'hasPart', 'bay'].edge_index = substation_bay_edge_index
    data['bay', 'hasPart', 'module'].edge_index = bay_module_edge_index
    data['module', 'directlyConnectedTo', 'module'].edge_index = module_module_edge_index

    return data