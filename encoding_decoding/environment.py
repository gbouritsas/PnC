from torch_geometric.utils import to_undirected
from torch_geometric.data import Batch
import torch
from encoding_decoding.encoding_costs import *
from utils.utils_subgraphs import induced_subgraph, compute_degrees, remove_overlapping_subgraphs

# epsilon = 1e-5
epsilon = 1e-7

# up to 15 nodes
num_connected_graphs = [1, 1, 2, 6, 21, 112, 853, 11117,
                        261080, 11716571, 1006700565, 164059830476, 50335907869219,
                        29003487462848061]
#31397381142761241960]
log_univ_size = torch.log2(torch.tensor(num_connected_graphs))

class CompressionEnvironment:
    
    def __init__(self, **kwargs):
    
        print("Preparing compression environment... ")

        self.device = kwargs['device'] if 'device' in kwargs else torch.device('cpu')
        for arg in ['directed', 'isomorphism_module',
                    # ENCODING SCHEMES
                    # dictionary
                    'dictionary_encoding', 'num_nodes_atom_encoding', 'num_edges_atom_encoding', 'adj_matrix_atom_encoding',
                    # dictionary and non-dictionary subgraphs
                    'dict_subgraphs_encoding', 'num_nodes_encoding', 'num_edges_encoding', 'adj_matrix_encoding',
                    # cuts
                    'cut_encoding', 'cut_size_encoding', 'cut_edges_encoding',
                    # attributes
                    'node_attr_encoding', 'edge_attr_encoding',
                    # baseline encoding
                    'num_nodes_baseline_encoding', 'num_edges_baseline_encoding', 'adj_matrix_baseline_encoding',
                     # ENCODING CONSTANTS
                     # dictionary encoding constants
                    'n_h_max_dict', 'n_h_min_dict',
                    # graph encoding constants
                    'c_max', 'n_max', 'e_max', 'd_max', 'b_min',
                    # precision for floats/fixed precision encoding,
                    'precision',
                    # fixed or adaptive dictionary
                    'universe_type']:
            setattr(self, arg, kwargs[arg])
        for arg in ['e_h_max_dict', 'd_h_max_dict']:
            setattr(self, arg, None)
        # attributes
        self.node_attr_unique_values = None if kwargs['node_attr_unique_values'] is None \
            else torch.tensor(kwargs['node_attr_unique_values'], device=self.device)
        self.edge_attr_unique_values = None if kwargs['edge_attr_unique_values'] is None \
            else torch.tensor(kwargs['edge_attr_unique_values'], device=self.device)

        # initialise dictionary if there is prior knowledge about the containing atoms
        self.max_dict_size = max(kwargs['max_dict_size'], len(kwargs['dictionary']))
        # empirical estimation of the atom probabilities: Important to accelerate graph isomorphism checks
        self.empirical_atom_probs = torch.ones((self.max_dict_size,), device=self.device) / self.max_dict_size
        self.empirical_atom_freqs = torch.ones((self.max_dict_size,), device=self.device)
        self.ema_coeff = kwargs['ema_coeff'] if 'ema_coeff' in kwargs else None # moving average coefficient
        self.dictionary, self.dictionary_num_vertices, self.dictionary_num_edges = [], [], []
        self.atom_costs = -1 * torch.ones((self.max_dict_size,), device=self.device)
        # update dictionary if initial atoms are given (in graph-tool format)
        if len(kwargs['dictionary'])!=0:
            edge_lists, n_h_s, e_h_s, atoms = [], [], [], []
            for i in range(len(kwargs['dictionary'])):
                n_h_s.append(kwargs['dictionary'][i].num_vertices())
                e_h_s.append(kwargs['dictionary'][i].num_edges())
                edge_lists.append(to_undirected(
                    torch.tensor(kwargs['dictionary'][i].get_edges(), device=self.device).transpose(1,0), num_nodes=n_h_s[-1]) if not self.directed
                                  else torch.tensor(kwargs['dictionary'][i].get_edges(), device=self.device).transpose(1,0))
                # initial attributed universe is not implemented
                node_attr_h = None #if self.node_attr_encoding is None else kwargs['dictionary'][i].node_attrs()
                edge_attr_h = None #if self.edge_attr_encoding is None else kwargs['dictionary'][i].edge_attrs()
                atoms.append(self.isomorphism_module.convert_graph(E=edge_lists[-1],
                                                                        n=n_h_s[-1],
                                                                        node_attrs=node_attr_h,
                                                                        edge_attrs=edge_attr_h,
                                                                        node_attr_dims=self.isomorphism_module.node_attr_dims,
                                                                        edge_attr_dims=self.isomorphism_module.edge_attr_dims))
            self.update_dict_atoms(atoms, n_h_s, e_h_s)

        # environment transition function between states
        # (needed only if the compressor can be treated as and MDP and
        # the description length can be rewritten as cumulative reward)
        self.step = self.subgraph_removal

    def map_to_dict(self, num_vertices, num_edges, edge_index,
                    dictionary_num_vertices, dictionary_num_edges, dictionary, query_inds,
                    directed=False, attr_mapping=None, node_features=None, edge_features=None):
        # attributes
        if self.isomorphism_module.node_attr_dims is not None or self.isomorphism_module.edge_attr_dims is not None:
            node_attrs, edge_attrs = attr_mapping.map(node_features, edge_features)
        else:
            node_attrs, edge_attrs = None, None
        # convert graph to the required format
        G = self.isomorphism_module.convert_graph(E=edge_index,
                                             n=num_vertices,
                                             directed=directed,
                                             node_attrs=node_attrs,
                                             edge_attrs=edge_attrs,
                                             node_attr_dims=self.isomorphism_module.node_attr_dims,
                                             edge_attr_dims=self.isomorphism_module.edge_attr_dims)
        found_atom_index = -1
        for atom_index in query_inds:
            if dictionary_num_vertices[atom_index] != num_vertices or \
                    dictionary_num_edges[atom_index] != num_edges:
                continue
            try:
                are_iso = self.isomorphism_module.match(G, dictionary[atom_index])
            except Exception as e:
                print('Timeout during isomorphism!')
                are_iso = False
            if are_iso:
                found_atom_index = atom_index
                break
        return found_atom_index, G

    def update_dict_atoms(self, atoms, n_h_s, e_h_s):
        self.dictionary += atoms
        self.dictionary_num_vertices += n_h_s
        self.dictionary_num_edges += e_h_s
        return

    def estimate_atom_freqs_probs(self, atom_indices):

        estimated_atom_freqs = torch.bincount(atom_indices, minlength=len(self.dictionary))
        estimated_atom_probs = estimated_atom_freqs / estimated_atom_freqs.sum()
        return estimated_atom_freqs, estimated_atom_probs

    def update_empirical_atom_freqs_probs(self, estimated_atom_freqs, estimated_atom_probs, save_update=True):
        empirical_atom_freqs = (1 - self.ema_coeff) * self.empirical_atom_freqs[0:len(self.dictionary)] \
                               + self.ema_coeff * estimated_atom_freqs
        empirical_atom_probs = (1 - self.ema_coeff) * self.empirical_atom_probs[0:len(self.dictionary)] \
                               + self.ema_coeff * estimated_atom_probs
        if save_update:
            self.empirical_atom_freqs[0:len(self.dictionary)] = empirical_atom_freqs
            self.empirical_atom_probs[0:len(self.dictionary)] = empirical_atom_probs
        return empirical_atom_freqs, empirical_atom_probs

    def sorted_atoms_in_dict(self, x_a=None):
        if x_a is None:
            sorted_probs_inds = torch.argsort(self.empirical_atom_probs[0:len(self.dictionary)], descending=True)
        else:
            keep_inds = torch.where(x_a[0:len(self.dictionary)] > 0.5)[0]
            sorted_probs_inds = torch.argsort(self.empirical_atom_probs[keep_inds], descending=True)
            sorted_probs_inds = keep_inds[sorted_probs_inds]
        return sorted_probs_inds.tolist()


    def baseline_graph_cost(self, n, e):
        return compute_cost_graph(self.n_max, n, e,
                                  self.num_nodes_baseline_encoding,
                                  self.num_edges_baseline_encoding, 
                                  self.adj_matrix_baseline_encoding, 
                                  self.e_max,
                                  self.d_max, 
                                  self.directed,
                                  self.precision,
                                  self.node_attr_encoding,
                                  self.node_attr_unique_values,
                                  self.edge_attr_encoding,
                                  self.edge_attr_unique_values)
    
    
    def graph_cost(self, n, e):
        return compute_cost_graph(self.n_max, n, e,
                                  self.num_nodes_encoding,
                                  self.num_edges_encoding, 
                                  self.adj_matrix_encoding, 
                                  self.e_max,
                                  self.d_max, 
                                  self.directed,
                                  self.precision,
                                  self.node_attr_encoding,
                                  self.node_attr_unique_values,
                                  self.edge_attr_encoding,
                                  self.edge_attr_unique_values)
    
    def compute_atom_costs(self):
        mask_to_compute = self.atom_costs[0:len(self.dictionary)] == -1
        if mask_to_compute.sum()!=0:
            if self.dictionary_encoding == 'graphs':
                n = torch.tensor(self.dictionary_num_vertices, device=self.device)[mask_to_compute]
                e = torch.tensor(self.dictionary_num_edges, device=self.device)[mask_to_compute]
                # in the dictionary we do not allow the empty atom,
                # hence the unique values for the number of vertices are {1,..., n_h_max_dict}
                self.atom_costs[0:len(self.dictionary)][mask_to_compute] = compute_cost_graph(
                    self.n_h_max_dict - self.n_h_min_dict, n, e,
                    self.num_nodes_atom_encoding,
                    self.num_edges_atom_encoding,
                    self.adj_matrix_atom_encoding,
                    self.e_h_max_dict,
                    self.d_h_max_dict,
                    self.directed,
                    self.precision,
                    self.node_attr_encoding,
                    self.node_attr_unique_values,
                    self.edge_attr_encoding,
                    self.edge_attr_unique_values)
            elif self.dictionary_encoding == 'isomorphism_classes':
                self.atom_costs[0:len(self.dictionary)][mask_to_compute] = (log_univ_size[self.n_h_max_dict - 1]).to(self.device)
            else:
                raise NotImplementedError("Dictionary encoding {} is not currently supported.".format(self.dictionary_encoding))

        return self.atom_costs[0:len(self.dictionary)]

    def cut_cost(self, c_ij, n_h_ij, e_h_ij, b=None):
        n_h_i, n_h_j = n_h_ij[0], n_h_ij[1]
        e_h_i, e_h_j = e_h_ij[0], e_h_ij[1]
        return compute_cost_cut(c_ij, n_h_i, n_h_j,
                         self.cut_encoding,
                         self.cut_size_encoding,
                         self.cut_edges_encoding,
                         b,
                         e_h_i,
                         e_h_j,
                         self.c_max,
                         self.d_max,
                         self.directed,
                         self.precision,
                         self.edge_attr_encoding,
                         self.edge_attr_unique_values)


    def compute_dl(self,
                   subgraphs,
                   x_a,
                   b_probs,
                   delta_prob,
                   atom_probs,
                   cut_size_probs,
                   log_probs=None):

        atom_indices = subgraphs['atom_indices']
        if x_a is not None:
            x_h = x_a[atom_indices]
            x_h[atom_indices == -1] = 0.0
        else:
            x_h = atom_indices != -1

        # num blocks
        b = subgraphs['b']
        b_dict = x_h.sum(1)

        # account for numerical errors
        if delta_prob == 0:
            delta_prob += epsilon
        elif delta_prob == 1:
            delta_prob -= epsilon
        current_b_probs = b_probs[b - self.b_min]
        current_b_probs[current_b_probs==0] = epsilon

        # number of dict and non-dict subgraphs
        cost_b = - torch_log_binom(b, b_dict) \
                 - b_dict * torch.log2(1 - delta_prob) \
                 - (b - b_dict) * torch.log2(delta_prob) \
                 - torch.log2(current_b_probs)
        # dict subgraphs
        b_a = torch.stack([torch.histc(atom_indices[i],
                                       bins=len(self.dictionary) + 1,
                                       min=-1, max=len(self.dictionary)) for i in range(len(atom_indices))])
        b_a = b_a[:, 1:] * x_a[0:len(self.dictionary)]

        assert ((b_dict - b_a.sum(1)).abs()<epsilon).all(), "assertion error in b_dict"
        cost_H_dict = compute_cost_dictionary_subgraphs(b_a,
                                                        atom_probs[0:len(self.dictionary)],
                                                        self.dict_subgraphs_encoding)


        # non dict subgraphs
        n_h, e_h, = subgraphs['n_h'], subgraphs['e_h']
        cost_H_null = (1 - x_h) * self.graph_cost(n_h, e_h)
        cost_H_null[n_h == 0] = 0
        cost_H_null = cost_H_null.sum(1)

        # cuts
        c_ij, n_h_ij, e_h_ij = subgraphs['c_ij'], subgraphs['n_h_ij'], subgraphs['e_h_ij']
        cost_C = self.cut_cost(c_ij, n_h_ij, e_h_ij, b)


        cost_G = cost_b + cost_H_dict + cost_H_null + cost_C
        assert not torch.isnan(cost_G).any(), "numerical error - probably a probability is equal to 0"
        # init graph
        if subgraphs['n_0'] is not None and subgraphs['e_0'] is not None:
            baseline = self.baseline_graph_cost(subgraphs['n_0'], subgraphs['e_0'])
        else:
            baseline = None
        log_probs = log_probs.sum(1) if log_probs is not None else None
        cost_terms = {'b':cost_b, 'H_dict': cost_H_dict, 'H_null': cost_H_null, 'C': cost_C}
        return cost_G, baseline, log_probs, cost_terms

    def subgraph_removal(self,  batched_graph, subgraphs, n_h_s, e_h_s, c_s,
                         num_nodes=None, remaining_nodes=None, relabel_nodes=None, candidate_subgraphs=True):
        
        if relabel_nodes is None:
            relabel_nodes=False
        
        num_nodes = int(batched_graph.graph_size.sum()) if num_nodes is None else int(num_nodes) #.sum() needed for batches
        batched_subgraph = torch.cat(subgraphs)

        # check if this is slow + check if the output is always ordered + check if I need tensor here
        remaining_nodes = set(range(num_nodes)) if remaining_nodes is None else remaining_nodes
        remaining_nodes = list(remaining_nodes.difference(set(batched_subgraph.tolist())))
                              
        if hasattr(batched_graph, 'edge_features'):
            edge_index, edge_features, relabelling = induced_subgraph(remaining_nodes,
                                                          batched_graph.edge_index, 
                                                          edge_attr=batched_graph.edge_features, 
                                                          relabel_nodes=relabel_nodes,
                                                          num_nodes=num_nodes)
        else:
            edge_index, _, relabelling = induced_subgraph(remaining_nodes, 
                                                          batched_graph.edge_index, 
                                                          edge_attr=None, 
                                                          relabel_nodes=relabel_nodes,
                                                          num_nodes=num_nodes)  
            
           
                              

        ##### make sure to change all the attributes appropriately 
        ##### check if this function is slow. Maybe I shouldn't create a new object?
                              
        new_batched_graph = Batch()
        new_num_nodes = len(remaining_nodes) if relabel_nodes else num_nodes

        setattr(new_batched_graph, 'edge_index', edge_index)
        if hasattr(batched_graph, 'graph_size'):
            new_graph_size = batched_graph.graph_size - n_h_s
            setattr(new_batched_graph, 'graph_size', new_graph_size)
            
            
        if hasattr(batched_graph, 'edge_size') and c_s is not None:
            new_edge_size  = batched_graph.edge_size - c_s[0] - e_h_s if not self.directed else \
                                batched_graph.edge_size - c_s[0] - c_s[1] - e_h_s
            setattr(new_batched_graph, 'edge_size', new_edge_size)

        if hasattr(batched_graph, 'edge_features'):
            setattr(new_batched_graph, 'edge_features', edge_features)

        if candidate_subgraphs:
            new_batched_graph = remove_overlapping_subgraphs(batched_graph,
                                                             new_batched_graph, 
                                                             remaining_nodes, 
                                                             num_nodes,
                                                             relabelling, 
                                                             len(self.dictionary),
                                                             relabel_nodes=relabel_nodes)

        if hasattr(batched_graph, 'degrees'):
            # this might be slow, but we need it to compute the reward either way
            degrees =  compute_degrees(new_batched_graph.edge_index, new_num_nodes, self.directed)
            setattr(new_batched_graph, 'degrees', degrees)
        if hasattr(batched_graph, 'x'):
            x = batched_graph.x[remaining_nodes] if relabel_nodes else batched_graph.x
            setattr(new_batched_graph, 'x', x) 
        if hasattr(batched_graph, 'batch'):
            batch = batched_graph.batch[remaining_nodes] if relabel_nodes else batched_graph.batch
            setattr(new_batched_graph, 'batch', batch) 
            
        return new_batched_graph, remaining_nodes, relabelling
