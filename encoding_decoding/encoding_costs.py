import torch
from utils.combinatorics import torch_log_binom, torch_log_factorial

##### can we obtain better upper bounds?????? #######
def num_edges_upper_bound(n, e_max=None, d_max=None, directed=False):
    
    ### for self-loops
    # e_max = min(n_tau_1**2, self.args['e_max']) if args['e_max'] is not None else n_tau_1**2
    
    if directed:
        e_max = torch.clamp(n**2 - n, max=e_max) if e_max is not None else n**2 - n # remove self loops from the upper bound
        if d_max is not None:
            e_max = torch.min(n * min(d_max), e_max)
    else:
        e_max = torch.clamp(n*(n-1)/2, max=e_max) if e_max is not None else n*(n-1)/2
        if d_max is not None:
            e_max = torch.min(torch.floor(n * d_max[0]/2), e_max)
    return e_max
        
def cut_size_upper_bound(k_1, k_2, e_1=None, e_2=None, c_max=None, d_max=None, directed=False):
    if d_max is not None and e_1 is not None and e_2 is not None:
        if directed:
            d_max_out, d_max_in = d_max
            c_max_out = torch.clamp(k_1 * d_max_out - e_1, max=c_max[0]) if c_max is not None else k_1 * d_max_out - e_1
            c_max_in = torch.clamp(k_2 * d_max_in - e_2, max=c_max[1]) if c_max is not None else k_2 * d_max_in - e_2
            c_max = torch.min(torch.min(c_max_out, c_max_in), k_1 * k_2)
        else:
            c_max_curr = torch.min(torch.min(k_1 * d_max[0] - 2 * e_1, k_2 * d_max[0] - 2 * e_2), k_1 * k_2)
            c_max = [torch.clamp(c_max_curr, max=c_max[0]) if c_max is not None else c_max_curr]
    else:
        c_max = torch.clamp(k_1 * k_2, max=c_max[0]) if c_max is not None else k_1 * k_2

    return c_max


def compute_cost_scalar(scalar_encoding, values, num_bits=None):
    if scalar_encoding == 'fixed_precision':
        scalar_cost = torch.ones_like(values) * num_bits
    elif scalar_encoding == 'uniform':
        unique_values = values
        scalar_cost = torch.log2(unique_values)
    elif scalar_encoding == 'entropy':
        data_probs = values
        scalar_cost = -torch.log2(data_probs)
    else:
        raise NotImplementedError("Encoding {} is not currently supported for scalar values.".format(scalar_encoding))
            
    return scalar_cost

def compute_cost_adj_matrix(adj_matrix_encoding, e, n, num_bits=None, directed=False):
    if adj_matrix_encoding == 'fixed_precision':
        adj_matrix_cost = 2 * e * num_bits
    elif adj_matrix_encoding == 'uniform':
        unique_values = n**2 - n if directed else n*(n-1)/2
        adj_matrix_cost = unique_values
    elif adj_matrix_encoding == 'edge_list':
        unique_values = n ** 2 - n if directed else n * (n - 1) / 2
        adj_matrix_cost = torch.log2(unique_values) * e
    elif adj_matrix_encoding == 'erdos_renyi':
        unique_values = n**2 - n if directed else n*(n-1)/2
        adj_matrix_cost = torch_log_binom(unique_values, e)
    else:
        raise NotImplementedError("Encoding {} is not currently supported for scalar values.".format(adj_matrix_encoding))

    return adj_matrix_cost


def compute_cost_graph(n_max, n, e,
                       num_nodes_encoding,
                       num_edges_encoding, 
                       adj_matrix_encoding, 
                       e_max=None,
                       d_max=None, 
                       directed=False, 
                       precision=None,
                       node_attr_encoding=None,
                       node_attr_unique_values=None,
                       edge_attr_encoding=None,
                       edge_attr_unique_values=None):

    # num nodes cost
    num_nodes_cost = compute_cost_scalar(num_nodes_encoding, torch.ones_like(n) * (n_max+1), precision)
    # num edges cost
    e_max = num_edges_upper_bound(n, e_max, d_max, directed)
    num_edges_cost = compute_cost_scalar(num_edges_encoding, e_max + 1, precision)
    # adjacency matrix cost
    adj_matrix_cost = compute_cost_adj_matrix(adj_matrix_encoding, e, n, precision, directed)
    if adj_matrix_encoding == 'uniform':
        graph_cost = num_nodes_cost + adj_matrix_cost
    else:
        graph_cost = num_nodes_cost + num_edges_cost + adj_matrix_cost

    if node_attr_encoding is not None:
        node_attr_cost = compute_cost_attr(node_attr_encoding, n, node_attr_unique_values, precision)
    else:
        node_attr_cost = 0
    if edge_attr_encoding is not None:
        edge_attr_cost = compute_cost_attr(edge_attr_encoding, e, edge_attr_unique_values, precision)
    else:
        edge_attr_cost = 0

    return graph_cost + node_attr_cost + edge_attr_cost


def compute_cost_cut_edges(cut_edges_encoding, c, k_1, k_2, num_bits=None):
    if cut_edges_encoding == 'fixed_precision':
        cut_edges_cost = num_bits * 2 * c
    elif cut_edges_encoding == 'edge_list':
        cut_edges_cost = c * (torch.log2(k_1) + torch.log2(k_2))
    elif cut_edges_encoding == 'erdos_renyi':
        cut_edges_cost = torch_log_binom(k_1 * k_2, c)
    else:
        raise NotImplementedError("Encoding {} is not currently supported for scalar values.".format(cut_edges_encoding))

    return cut_edges_cost

def compute_cost_cut(c, k_1, k_2,
                     cut_encoding,
                     cut_size_encoding,
                     cut_edges_encoding,
                     b=None,
                     e_1=None,
                     e_2=None,
                     c_max=None,
                     d_max=None, 
                     directed=False, 
                     precision=None,
                     edge_attr_encoding=None,
                     edge_attr_unique_values=None):


    # # cut size cost abd cut edges cost
    if cut_encoding == 'uniform':
        cut_cost = (k_1 * k_2).sum(1)
    elif cut_encoding == 'independent':
        c_max = cut_size_upper_bound(k_1, k_2, e_1, e_2, c_max, d_max, directed)
        cut_size_cost = compute_cost_scalar(cut_size_encoding, c_max + 1, precision)
        cut_edges_cost = compute_cost_cut_edges(cut_edges_encoding, c, k_1, k_2, precision)
        cut_cost = (cut_size_cost + cut_edges_cost).sum(-1)
    elif cut_encoding == 'joint':
        cut_size_cost = torch.log2((k_1 * k_2).sum(1) + 1) + torch_log_binom(b * (b - 1) / 2 + c.sum(1) - 1, c.sum(1))
        cut_edges_cost = compute_cost_cut_edges(cut_edges_encoding, c, k_1, k_2, precision)
        cut_cost = cut_size_cost + cut_edges_cost.sum(-1)
    else:
        raise NotImplementedError("Encoding {} is not currently supported.".format(cut_encoding))

    if edge_attr_encoding is not None:
        edge_attr_cost = compute_cost_attr(edge_attr_encoding, c, edge_attr_unique_values, precision).sum(-1)
    else:
        edge_attr_cost = 0

    return cut_cost + edge_attr_cost


def compute_cost_dictionary_subgraphs(b_a, atom_probs, dict_subgraphs_encoding):
    # dict subgraphs
    if dict_subgraphs_encoding == 'multinomial':
        b_dict = b_a.sum(1)
        dict_subgraphs_cost = - torch_log_factorial(b_dict)\
                              + torch_log_factorial(b_a).sum(1) \
                              - (b_a[:,atom_probs!=0] * torch.log2(atom_probs[atom_probs!=0])).sum(1)
    elif dict_subgraphs_encoding == 'categorical':
        dict_subgraphs_cost = - (b_a[:,atom_probs!=0] * torch.log2(atom_probs[atom_probs!=0])).sum(1)
    else:
        raise NotImplementedError("Encoding {} is not currently supported for scalar values.".format(dict_subgraphs_encoding))
    return dict_subgraphs_cost

def compute_cost_attr(attr_encoding, n, values, num_bits=None):
    if attr_encoding == 'fixed_precision':
        attr_cost = (torch.ones_like(values[(None,) * n.dim()]) * n.unsqueeze(-1)).sum(-1) * num_bits
    elif attr_encoding  == 'uniform':
        unique_values = values
        attr_cost = (torch.log2(unique_values[(None,)*n.dim()]) * n.unsqueeze(-1)).sum(-1)
    elif attr_encoding == 'entropy':
        raise NotImplementedError('entropy coding of attributes has not been tested')
        # Not tested yet
        data_probs = values
        attr_cost = -torch.log2(data_probs).sum(-1)
    else:
        raise NotImplementedError("Encoding {} is not currently supported for scalar values.".format(attr_encoding))
    return attr_cost

