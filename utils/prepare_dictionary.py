import os
import types
import pickle
from utils.misc import isnotebook
if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.stats as gt_stats
import graph_tool.topology as gt_topology
import graph_tool.clustering as gt_clustering
import graph_tool.generation as gt_generation
from utils.conversions import convert_to_gt

def unique_non_isomorphic(H_set):
    H_unique = []
    for H in H_set:
        found = False
        for H_saved in H_unique:
            if H.num_vertices() != H_saved.num_vertices() or H.num_edges() != H_saved.num_edges():
                # avoid running isomorphism routine if num vertices/num edges is different
                continue
            iso = True if H.num_edges() == 0 and H.num_vertices() == 1 else \
                gt_topology.isomorphism(H, H_saved)
            if iso:
                found = True
                break
        if not found:
            H_unique.append(H)
    return H_unique

def get_motifs(k_min, k_max, graphs_ptg, directed=False):
    #n_shuffles = 100
    motif_num_vertices_list = list(range(k_min, k_max+1))
    H_dictionary = []
    counts = []
    # add single nodes and single edges
    H_dictionary += [gt_generation.complete_graph(1), gt_generation.complete_graph(2)]
    counts += [0,0]
    for i in tqdm(range(len(graphs_ptg))):
        G_edge_index = graphs_ptg[i].edge_index.transpose(1,0).tolist()
        G_gt = gt.Graph(directed=directed)
        G_gt.add_edge_list(G_edge_index)
        gt_stats.remove_self_loops(G_gt)
        gt_stats.remove_parallel_edges(G_gt)
        for motif_num_vertices in motif_num_vertices_list:
            motifs_k, counts_k = gt_clustering.motifs(G_gt, motif_num_vertices)
            for motif, count in zip(motifs_k, counts_k):
                found=False
                for H_index, H in enumerate(H_dictionary):
                    if H.num_vertices() != motif.num_vertices() or H.num_edges() != motif.num_edges():
                        # avoid running isomorphism routine if num vertices/num edges is different
                        continue
                    iso = True if H.num_edges() == 0 and H.num_vertices()==1 else \
                        gt_topology.isomorphism(H, motif)
                    if iso:
                        counts[H_index] += count
                        found = True
                        break
                if not found:
                    H_dictionary.append(motif)
                    counts += [count]
    counts = np.array(counts)
    H_dictionary = list(np.array(H_dictionary)[np.argsort(-counts)])
    counts = counts[np.argsort(-counts)]
    counts = counts/counts.sum()
    return H_dictionary, counts

def get_custom_edge_list(ks, substructure_type=None, filename=None):
    '''
        Instantiates a list of `edge_list`s representing substructures
        of type `substructure_type` with sizes specified by `ks`.
    ''' 
    if substructure_type is None and filename is None:
        raise ValueError('You must specify either a type or a filename where to read substructures from.')
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))
    return edge_lists


def prepare_dictionary(args, path=None, graphs_ptg=None, split_folder=None):
    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user
    H_set_gt = []
    edge_lists_all = []
    for i, atom_type in enumerate(args['atom_types']):
        if atom_type in ['cycle_graph',
                         'path_graph',
                         'complete_graph',
                         'binomial_tree',
                         'star_graph',
                         'nonisomorphic_trees']:
            k_min = 2 if atom_type == 'star_graph' else 1
            k_max = args['k'][i]
            edge_lists = get_custom_edge_list(list(range(k_min, k_max + 1)), substructure_type=atom_type)
        elif atom_type in ['cycle_graph_chosen_k',
                           'path_graph_chosen_k', 
                           'complete_graph_chosen_k',
                           'binomial_tree_chosen_k',
                           'star_graph_chosen_k',
                           'nonisomorphic_trees_chosen_k']:
            edge_lists = get_custom_edge_list([args['k'][i]], substructure_type=atom_type.replace('_chosen_k',''))
        elif atom_type == 'all_simple_graphs':
            k_min = 2
            k_max = args['k'][i]
            filename = os.path.join(args['root_folder'], 'all_simple_graphs')
            edge_lists = get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)
        elif atom_type == 'all_simple_graphs_chosen_k':
            filename = os.path.join(args['root_folder'], 'all_simple_graphs')
            edge_lists = get_custom_edge_list([args['k'][i]], filename=filename)
        elif atom_type == 'diamond_graph':
            graph_nx = nx.diamond_graph()
            edge_lists = [list(graph_nx.edges)]
        elif atom_type == 'custom':
            assert args['custom_edge_lists'] is not None, "Custom edge lists must be provided."
            edge_lists = args['custom_edge_lists']
        elif atom_type == 'motifs':
            k_min = 3
            k_max = args['k'][i]
            # data_folder = os.path.join(path, 'processed', 'dictionaries')
            data_folder = os.path.join(path, 'processed', 'dictionaries', split_folder)
            motif_file = os.path.join(data_folder, 'motifs' + '_' + str(k_max) + '.pkl')
            if os.path.exists(motif_file):
                with open(motif_file, 'rb') as f:
                    H_set_gt, counts = pickle.load(f)
            else:
                H_set_gt, counts = get_motifs(k_min, k_max, graphs_ptg, directed=args['directed'])
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                with open(motif_file, 'wb') as f:
                    pickle.dump((H_set_gt, counts), f)
        else:
            raise NotImplementedError("Atom {} is not currently supported.".format(atom_type))
        if atom_type != 'motifs':
            edge_lists_all += edge_lists
    # convert to graph tool. Only necessary for subgraph isomorphism
    if len(edge_lists_all)!=0:
        H_set_gt += convert_to_gt(edge_lists_all, directed=args['directed'])
    H_set_gt = unique_non_isomorphic(H_set_gt)
        
    return H_set_gt
        
