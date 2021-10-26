from utils.misc import isnotebook
if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from utils.load_dataset import load_data, load_zinc_data, load_ogb_data, load_g6_graphs
from utils.utils_subgraphs import compute_degrees
import os
import torch_geometric.datasets as ptg_datasets

def unique_indices(num_unique, inverse):
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(num_unique).scatter_(0, inverse, perm)
    return perm

def generate_dataset(data_path, 
                     dataset_name,
                     directed):

    ### load and preprocess dataset

    dataset_family = os.path.split(os.path.split(data_path)[0])[1]
    if dataset_family == 'PPI':
        dataset_type = 'ptg'
        graphs = []
        start = 0
        if not os.path.exists(os.path.join(data_path, '10fold_idx')):
            os.makedirs(os.path.join(data_path, '10fold_idx'))
        for split in ['train', 'test', 'val']:
            graphs_temp = getattr(ptg_datasets, dataset_family)(data_path, split)
            graphs += [graphs_temp[i] for i in range(len(graphs_temp))]
            end = len(graphs)
            split_idx = list(range(start, end))
            filename = os.path.join(data_path, '10fold_idx', split+'_idx-{}.txt'.format(0))
            np.savetxt(filename, np.array(split_idx).astype(int), fmt='%d')
            start = end
        num_classes = graphs_temp.num_classes
        num_node_type, num_edge_type = None, None
    elif dataset_family == 'KarateClub':
        dataset_type = 'ptg'
        graphs = getattr(ptg_datasets, dataset_family)()
        num_classes = graphs.num_classes
        num_node_type, num_edge_type = None, None
    elif dataset_family == 'TUDataset':
        dataset_type = 'ptg'
        graphs =  getattr(ptg_datasets, dataset_family)(data_path, dataset_name, cleaned=True)
        num_classes = graphs.num_classes
        num_node_type, num_edge_type = None, None
    elif dataset_family == 'Amazon':
        dataset_type = 'ptg'
        graphs = getattr(ptg_datasets, dataset_family)(data_path, dataset_name)
        num_classes = graphs.num_classes
        num_node_type, num_edge_type = None, None
    elif dataset_family == 'Planetoid':
        dataset_type = 'ptg'
        graphs = getattr(ptg_datasets, dataset_family)(data_path, dataset_name)
        num_classes = graphs.num_classes
        num_node_type, num_edge_type = None, None
    elif 'ogb' in data_path:
        dataset_type = 'general'
        graphs, num_classes = load_ogb_data(data_path, dataset_name, False)
        num_node_type, num_edge_type = None, None
    elif dataset_name == 'ZINC':
        dataset_type = 'general'
        graphs, num_classes, num_node_type, num_edge_type = load_zinc_data(data_path, dataset_name, False)
    elif os.path.split(data_path)[-1] in ['SR_graphs', 'all_graphs']:
        dataset_type = 'general'
        graphs, num_classes = load_g6_graphs(data_path, dataset_name)
        num_node_type, num_edge_type = None, None
    else:
        dataset_type = 'general'
        graphs, num_classes = load_data(data_path, dataset_name, False)
        num_node_type, num_edge_type = None, None
        
    graphs_ptg = list()
    for i, data in tqdm(enumerate(graphs)):
        new_data = _prepare(data, directed, dataset_type, dataset_name)
        graphs_ptg.append(new_data)

    return graphs_ptg, num_classes, num_node_type, num_edge_type

# ------------------------------------------------------------------------
        
def _prepare(data, directed, dataset_type='ptg', dataset_name=None):
    new_data = Data()
    # nodes
    if dataset_type == 'ptg':
        if hasattr(data, 'x') and data.x is not None:
            num_nodes = data.x.shape[0]
            x = data.x
        else:
            num_nodes = data.num_nodes
            x = torch.ones((num_nodes,1))
        setattr(new_data, 'x', x)
    else:
        num_nodes = data.node_features.shape[0]
        setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'graph_size', float(num_nodes))
    # edges
    if dataset_type == 'ptg':
        num_edges = float(data.edge_index.shape[1]) if directed else data.edge_index.shape[1]/2
        edge_index = data.edge_index
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_features = data.edge_attr
        else:
            edge_features = None
    else:
        num_edges = float(data.edge_mat.shape[1]) if directed else data.edge_mat.shape[1]/2
        edge_index = data.edge_mat
        if hasattr(data, 'edge_features') and data.edge_features is not None:
            edge_features = data.edge_features
        else:
            edge_features = None
    setattr(new_data, 'edge_size', num_edges)
    
    # adjacency
    if edge_index.numel()!=0:
        # multi-edge graphs not allowed
        init_num_edges = edge_index.shape[1]
        edge_index, inverse = torch.unique(edge_index, dim=1, sorted=True, return_inverse=True)
        kept_inds = unique_indices(edge_index.shape[1], inverse)
        
        # warning messages
        if init_num_edges!=edge_index.shape[1]:
            print('Warning: detected duplicate edges')
        init_num_edges = edge_index.shape[1]
        
        if edge_features is not None:
            edge_features = edge_features[kept_inds]
            edge_index, edge_features = remove_self_loops(edge_index, edge_features)
        else:
            edge_index, _ = remove_self_loops(edge_index, None)
            
        # warning messages
        if init_num_edges!=edge_index.shape[1]:
            print('Warning: detected self loops')
    setattr(new_data, 'edge_index', edge_index)
    
    # edge features
    if edge_features is not None:
        setattr(new_data, 'edge_features', edge_features)  
    
    # degrees
    degrees = compute_degrees(edge_index, num_nodes, directed)
    setattr(new_data, 'degrees', degrees)
    
#     if regression or dataset_name in {'ogbg-molpcba', 'ogbg-molhiv', 'ZINC'}:
#         setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).float())
#     else:
#         setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())

    return new_data

# --------------------------------------------------------------------------------------