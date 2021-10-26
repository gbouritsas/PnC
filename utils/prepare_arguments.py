from torch_geometric.utils import degree
import math

def prepare_environment_args(args,
                             graphs_ptg, 
                             dictionary,
                             device,
                             isomorphism_module,
                             node_attr_unique_values,
                             edge_attr_unique_values):
    
    # n_max
    if args['n_max'] is not None:
        args['n_max'] = max(args['n_max'], max([graph.graph_size for graph in graphs_ptg]))
    else:
        args['n_max'] = max([graph.graph_size for graph in graphs_ptg])
    # e_max
    if args['e_max'] is not None:
        if args['directed']:
            args['e_max'] = max(args['e_max'], max([graph.edge_index.shape[1] for graph in graphs_ptg]))
        else:
            args['e_max'] = max(args['e_max'], int(max([graph.edge_index.shape[1] for graph in graphs_ptg])/2))
    if args['d_max'] is not None:
        if args['directed']:
            args['d_max'] = [max(args['d_max'][0], max([int(max(degree(graph.edge_index[0])).item()) for graph in graphs_ptg])),
                             max(args['d_max'][1], max([int(max(degree(graph.edge_index[1])).item()) for graph in graphs_ptg]))]
        else:
            args['d_max'] = [max(args['d_max'], max([int(max(degree(graph.edge_index[0])).item()) for graph in graphs_ptg]))]

    if args['n_h_max_dict'] is None or args['n_h_max_dict']==-1:
        args['n_h_max_dict'] = int(args['n_max'])
    if args['n_h_min_dict'] is None:
        args['n_h_min_dict'] = 1

    if args['n_h_max'] is None or args['n_h_max']==-1:
        args['n_h_max'] = int(args['n_max'])
    if args['n_h_min'] is None:
        args['n_h_min'] = 1

    #args['b_max'] = math.ceil(int(args['n_max'])/args['n_h_min'])
    args['b_max'] = int(args['n_max']) if args['b_max'] is None else max(args['b_max'], int(args['n_max']))
    args['b_min'] = math.ceil(int(args['n_max'])/args['n_h_max']) if args['b_min'] is None \
        else min(args['b_min'], math.ceil(int(args['n_max'])/args['n_h_max']))


    environment_args={}
    for key in ['directed',
                'universe_type', 'max_dict_size',
                'dictionary_encoding', 'num_nodes_atom_encoding', 'num_edges_atom_encoding', 'adj_matrix_atom_encoding',
                'dict_subgraphs_encoding',
                'num_nodes_encoding', 'num_edges_encoding', 'adj_matrix_encoding',
                'cut_encoding', 'cut_size_encoding', 'cut_edges_encoding',
                'node_attr_encoding', 'edge_attr_encoding',
                'num_nodes_baseline_encoding', 'num_edges_baseline_encoding', 'adj_matrix_baseline_encoding',
                'precision', 'n_max', 'e_max', 'd_max', 'c_max',
                'n_h_max_dict', 'n_h_min_dict', 'b_min',
                'ema_coeff']:
        environment_args[key] = args[key]
     
    environment_args['dictionary'] = dictionary
    environment_args['device'] = device
    environment_args['isomorphism_module'] = isomorphism_module
    environment_args['node_attr_unique_values'] = node_attr_unique_values
    environment_args['edge_attr_unique_values'] = edge_attr_unique_values
    return environment_args

def prepare_model_args(args, dictionary_size):
    print('Preparing model arguments....')
    model_args = {}
    # define if degree is going to be used as a feature and when (for each layer or only at initialization)
    if args['inject_degrees']:
        model_args['degree_as_tag'] = [args['degree_as_tag'] for _ in range(args['num_layers'])]
    else:
        model_args['degree_as_tag'] = [args['degree_as_tag']] + [False for _ in range(args['num_layers'] - 1)]
    # define if existing features are going to be retained when the degree is used as a feature
    model_args['retain_features'] = [args['retain_features']] + [True for _ in range(args['num_layers'] - 1)]
    # replicate d_out dimensions for the node/edge features and degree embeddings
    if args['d_out_node_embedding'] is None:
        model_args['d_out_node_embedding'] = args['d_out']
    if args['d_out_edge_embedding'] is None:
        model_args['d_out_edge_embedding'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        model_args['d_out_edge_embedding'] = [args['d_out_edge_embedding'] for _ in range(args['num_layers'])]
    if args['d_out_degree_embedding'] is None:
        model_args['d_out_degree_embedding'] = args['d_out']
    # replicate d_out dimensions if the rest are not defined (msg function, mlp hidden dimension, encoders, etc.)
    # and repeat hyperparams for every layer
    if args['d_msg'] == -1:
        model_args['d_msg'] = [None for _ in range(args['num_layers'])]
    elif args['d_msg'] is None:
        model_args['d_msg'] = [args['d_out'] for _ in range(args['num_layers'])]
    else:
        model_args['d_msg'] = [args['d_msg'] for _ in range(args['num_layers'])]
    if args['d_h'] is None:
        model_args['d_h'] = [[args['d_out']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
    else:
        model_args['d_h'] = [[args['d_h']] * (args['num_mlp_layers'] - 1) for _ in range(args['num_layers'])]
    model_args['train_eps'] = [args['train_eps'] for _ in range(args['num_layers'])]
    model_args['bn'] = [args['bn'] for _ in range(args['num_layers'])]
    if len(args['final_projection']) == 1:
        model_args['final_projection'] = [args['final_projection'][0] for _ in range(args['num_layers'])] + [True]
    model_args['dropout'] = [args['dropout'] for _ in range(args['num_layers'])] + [args['dropout']]
    # define output dimension based on the type of the partitioning algorithm
    if not args['candidate_subgraphs']:
        if args['partitioning_algorithm'] == 'subgraph_selection':
            # categorical on possible subgraph sizes + categorical on the vertices
            model_args['out_graph_features'] = args['n_h_max'] - args['n_h_min'] + 1 \
                if (args['n_h_max'] is not None) and (args['n_h_min'] is not None) else None
            model_args['out_node_features'] = 1
        elif args['partitioning_algorithm'] == 'subgraph_selection_w_termination':
            # bernoulli (continue or terminate) + categorical on the vertices
            model_args['out_graph_features'] = 1
            model_args['out_node_features'] = 1
        elif args['partitioning_algorithm'] =='contraction':
            # categorical on the number of clusters (or bernoulli - terminate/continue) +
            # categorical on the edges (contract until desired num clusters is reached)
            model_args['out_graph_features'] = args['n_h_max'] - args['n_h_min'] + 1 \
                if (args['n_h_max'] is not None) and (args['n_h_min'] is not None) else None
            model_args['out_edge_features'] = 1
            model_args['directed'] = args['directed']
        elif args['partitioning_algorithm'] =='clustering':
            # categorical on the number of clusters + clustering algorithm in the latent space
            model_args['out_graph_features'] = args['n_h_max'] - args['n_h_min'] + 1 \
                if (args['n_h_max'] is not None) and (args['n_h_min'] is not None) else None
            model_args['out_node_features'] = args['d_out']
            # for iterative clustering in the latent space
            model_args['clustering_iters'] = args['clustering_iters']
            model_args['clustering_temp'] =  args['clustering_temp']
        else:
            raise NotImplementedError('partitioning algorithm {} not implemented'.format(args['partitioning_algorithm']))
    else:
        # independent set problem (here we sample subgraphs from the candidate set iteratively)
        model_args['out_subgraph_features'] = 1
        model_args['dictionary_size'] = dictionary_size
        if args['degree_as_tag_pool'] is None:
            model_args['degree_as_tag_pool'] = args['degree_as_tag'][0]
        if args['retain_features_pool'] is None:
            model_args['retain_features_pool'] = args['retain_features'][0]
        if args['f_fn_type'] is None:
            model_args['f_fn_type'] = 'general'
        if args['phi_fn_type'] is None:
            model_args['phi_fn_type'] = 'general'
        if args['f_d_out'] is None:
            model_args['f_d_out'] = args['d_out']
        if args['d_h_pool'] is None:
            model_args['d_h_pool'] = args['d_h'][0]
        if args['aggr_pool'] is None:
            model_args['aggr_pool'] = args['aggr']
    # repeat width for every layer
    model_args['d_out'] = [args['d_out'] for _ in range(args['num_layers'])]
    for key in ['input_node_embedding', 'edge_embedding', 'degree_embedding',
                'inject_edge_features', 'multi_embedding_aggr',
                'aggr', 'flow', 'extend_dims',
                'activation_mlp', 'bn_mlp', 'activation',
                'model_name','aggr_fn', 'final_projection_layer','readout']:
        model_args[key] = args[key]

    return model_args
