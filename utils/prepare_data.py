import os
import random
from copy import deepcopy
import torch
from torch_geometric.data import DataLoader
from utils.attributes import AttrMapping
from utils.generate_dataset import generate_dataset
from utils.feature_encoding import encode_features
from utils.load_dataset import separate_data, separate_data_given_split

def prepare_dataset(path, 
                    dataset, 
                    name,
                    directed=False,
                    H_set = None,
                    multiprocessing=False,
                    num_processes=1,
                    candidate_subgraphs=False):
    
    if dataset in ['KarateClub', 'PPI',
                   'Planetoid', 'Amazon', 'TUDataset',
                   'bioinformatics', 'social',
                   'chemical', 'ogb',
                   'SR_graphs', 'all_graphs']:
        data_folder = os.path.join(path, 'processed', 'dictionaries')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            
        graphs_ptg_file = os.path.join(data_folder, 'directed_graphs_ptg.pt') if directed \
                        else os.path.join(data_folder, 'graphs_ptg.pt')
        if not os.path.exists(graphs_ptg_file):
            graphs_ptg, num_classes, num_node_type, num_edge_type = generate_dataset(path,
                                                                                     name,
                                                                                     directed)
            torch.save((graphs_ptg, num_classes), graphs_ptg_file)
            if num_node_type is not None:
                torch.save((num_node_type, num_edge_type), os.path.join(data_folder, 'num_feature_types.pt'))

        else:
            graphs_ptg, num_classes = torch.load(graphs_ptg_file)
            
        if candidate_subgraphs:
            #### precompute subgraphs
            from utils.subgraph_detection import detect_load_subgraphs
            graphs_ptg = detect_load_subgraphs(graphs_ptg, data_folder, H_set, directed, multiprocessing, num_processes)
    else:
        raise NotImplementedError("Dataset family {} is not currently supported.".format(dataset))


    return graphs_ptg

def prepape_input_features(args, graphs_ptg, path):


    ## ----------------------------------- node and edge feature dimensions
    if 'retain_features' in args and not args['retain_features']:
        for graph in graphs_ptg:
            graph.x = torch.ones((graph.x.shape[0],1))

    d_in_node_features = 1 if graphs_ptg[0].x.dim()==1 else graphs_ptg[0].x.shape[1]
    if hasattr(graphs_ptg[0], 'edge_features'):
        d_in_edge_features = 1 if graphs_ptg[0].edge_features.dim()==1 else graphs_ptg[0].edge_features.shape[1]
    else:
        d_in_edge_features = None

    if args['dataset'] in ['PPI', 'KarateClub', 'Planetoid', 'Amazon']:
        #'ppi' has continuous features - don't use it with attributes
        node_attr_unique_values, node_attr_dims = [2 for _ in range(d_in_node_features)], d_in_node_features
        edge_attr_unique_values, edge_attr_dims = None, None
        attr_mapping = AttrMapping(args['dataset_name'], 'integer', node_attr_dims, edge_attr_dims)
    elif args['dataset'] in ['TUDataset', 'social', 'bioinformatics']:
        node_attr_unique_values, node_attr_dims = [d_in_node_features], 1
        if d_in_edge_features is not None:
            edge_attr_unique_values, edge_attr_dims = [d_in_edge_features], 1
        else:
            edge_attr_unique_values, edge_attr_dims = None, None
        attr_mapping = AttrMapping(args['dataset_name'], 'one_hot', node_attr_dims, edge_attr_dims)
    elif args['dataset'] == 'chemical':
        node_attr_unique_values, edge_attr_unique_values = torch.load(os.path.join(path, 'processed/dictionaries', 'num_feature_types.pt'))
        node_attr_unique_values, edge_attr_unique_values = [node_attr_unique_values], [edge_attr_unique_values]
        attr_mapping = AttrMapping(args['dataset_name'], 'integer', 1, 1)
    elif args['dataset'] == 'ogb':
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
        node_attr_unique_values = get_atom_feature_dims()
        edge_attr_unique_values = get_bond_feature_dims()
        attr_mapping = AttrMapping(args['dataset_name'], 'integer', len(node_attr_unique_values), len(edge_attr_unique_values))
    else:
        raise NotImplementedError


    ## ----------------------------------- encode degrees (and possibly edge features and ids)

    degree_encoding = args['degree_encoding'] if ('degree_as_tag' in args and args['degree_as_tag']) else None
    print("Encoding degree features... \n", end='')
    encoding_parameters = {}
    graphs_ptg, degree_unique_values,_  = encode_features(graphs_ptg,
                                                          degree_encoding,
                                                          None,
                                                          **encoding_parameters)

    #degree_unique_values = degree_unique_values if args['d_max'] is None else [args['d_max'] + 1]
    in_features_dims_dict = {'d_in_node_features': d_in_node_features,
                             'node_attr_unique_values': node_attr_unique_values,
                             'd_in_edge_features': d_in_edge_features,
                             'edge_attr_unique_values': edge_attr_unique_values,
                             'degree_unique_values': degree_unique_values}
    
    return graphs_ptg, in_features_dims_dict, attr_mapping
    

def prepare_dataloaders(args,
                        graphs_ptg,
                        path,
                        fold_idx='',
                        candidate_subgraphs=True,
                        dictionary_size=0):
    graphs_ptg_modified = deepcopy(graphs_ptg)
    follow_batch = []
    if candidate_subgraphs:
        if hasattr(graphs_ptg[0], 'subgraph_detections'):
            for graph in graphs_ptg_modified:
                for i in range(dictionary_size):
                    subgraph_index = graph.subgraph_detections[i].transpose(1, 0) if graph.subgraph_detections[
                                                                                         i].numel() != 0 else torch.tensor([])
                    setattr(graph, 'subgraph_index_' + str(i), subgraph_index.long())
                del graph.subgraph_detections
        # check follow_batch
        for i in range(dictionary_size):
            follow_batch.append('subgraph_index_' + str(i))
    # split data into training/validation/test
    if args['split'] == 'random':  # use a random split
        split_folder = 'split_idx_random_' + str(args['split_seed'])
        if os.path.exists(os.path.join(path, split_folder)):
            dataset_train, dataset_test, dataset_val = separate_data_given_split(graphs_ptg_modified,
                                                                                 path,
                                                                                 fold_idx,
                                                                                 split_folder)
        else:
            os.makedirs(os.path.join(path, split_folder))
            dataset_train, dataset_test = separate_data(graphs_ptg_modified,
                                                        args['split_seed'],
                                                        fold_idx,
                                                        path,
                                                        split_folder)
        dataset_val = None
    elif args['split'] == 'given':  # use a precomputed split
        dataset_train, dataset_test, dataset_val = separate_data_given_split(graphs_ptg_modified,
                                                                             path,
                                                                             fold_idx)
    elif args['split'] == 'None':
        dataset_train, dataset_test, dataset_val = graphs_ptg_modified, None, None
    else:
        raise NotImplementedError('data split {} method not implemented'.format(args['split']))
    # instantiate data loaders
    loader_train = DataLoader(dataset_train,
                              batch_size=args['batch_size'],
                              shuffle=args['shuffle'],
                              worker_init_fn=random.seed(args['seed']),
                              num_workers=args['num_workers'],
                              follow_batch=follow_batch)
    if dataset_test is not None:
        loader_test = DataLoader(dataset_test,
                                 batch_size=args['batch_size'],
                                 shuffle=False,
                                 worker_init_fn=random.seed(args['seed']),
                                 num_workers=args['num_workers'],
                                 follow_batch=follow_batch)
    else:
        loader_test = None
    if dataset_val is not None:
        loader_val = DataLoader(dataset_val,
                                batch_size=args['batch_size'],
                                shuffle=False,
                                worker_init_fn=random.seed(args['seed']),
                                num_workers=args['num_workers'],
                                follow_batch=follow_batch)
    else:
        loader_val = None
    return loader_train, loader_test, loader_val
