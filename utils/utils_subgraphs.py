import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree

def repeat_batched_vector(batched_vec):
    num_graphs, T = batched_vec.shape
    matrix = batched_vec.repeat(1, T).view(num_graphs,T , T)
    matrix[torch.diag_embed(torch.ones(num_graphs, T)) == 1] = 0
    return matrix


def compute_combinations(batched_vec, directed=False):
    num_graphs, T = batched_vec.shape
    batched_matrix = repeat_batched_vector(batched_vec)
    if not directed:
        batched_vec_i = batched_matrix.transpose(2,1)[torch.triu(torch.ones_like(batched_matrix), diagonal=1) == 1].view(num_graphs, -1)
        batched_vec_j = batched_matrix[torch.triu(torch.ones_like(batched_matrix), diagonal=1) == 1].view(num_graphs, -1)
    else:
        batched_vec_i = batched_matrix.transpose(2,1).view(batched_matrix.shape[0], -1)
        batched_vec_j = batched_matrix.view(batched_matrix.shape[0], -1)
    return [batched_vec_i, batched_vec_j]


def compute_degrees(edge_index, num_nodes, directed):

    if edge_index.numel() == 0:
        degrees = torch.zeros((2, num_nodes), device=edge_index.device) if directed \
                    else torch.zeros((num_nodes,), device=edge_index.device)
    else:
        if directed:
            degrees_out = degree(edge_index[0], num_nodes)
            degrees_in = degree(edge_index[1], num_nodes)
            degrees = torch.stack((degrees_out, degrees_in))
        else:
            degrees = degree(edge_index[0], num_nodes)

    return degrees

def compute_cut_size_pairs(graph,
                           subgraphs,
                           directed=False):

    # for j in range(len(subgraphs)):
    #     subgraph_j = subgraphs[j]
    #     for i in range(0, j):
            # subgraph_i = subgraphs[i]
    cut_matrix = torch.zeros((len(subgraphs), len(subgraphs)), device=graph.edge_index.device)
    for i in range(len(subgraphs)):
        subgraph_i = subgraphs[i]
        if subgraph_i.numel()!=0:
            for j in range(i + 1, len(subgraphs)):
                subgraph_j = subgraphs[j]
                if subgraph_j.numel() != 0:
                    cut_matrix[i, j] = ((graph.edge_index[0, :] == subgraph_i.unsqueeze(1)).any(0)
                                          & (graph.edge_index[1, :] == subgraph_j.unsqueeze(1)).any(0)).sum()
                    if directed:
                        cut_matrix[j, i] = ((graph.edge_index[1, :] == subgraph_i.unsqueeze(1)).any(0)
                                              & (graph.edge_index[0, :] == subgraph_j.unsqueeze(1)).any(0)).sum()
    return cut_matrix


def compute_cut_size(graph,
                     subgraphs,
                     e_h_s,
                     num_nodes=None,
                     directed=False):
    device = e_h_s.device

    degrees = graph.degrees if hasattr(graph, 'degrees') else compute_degrees(graph.edge_index, num_nodes, directed)

    if directed:
        cut_size_out = torch.tensor([degrees[0][subgraph].sum() - e_h \
                                     for e_h, subgraph in zip(e_h_s, subgraphs)], device=device)
        cut_size_in = torch.tensor([degrees[1][subgraph].sum() - e_h \
                                    for e_h, subgraph in zip(e_h_s, subgraphs)], device=device)
        cut_size = [cut_size_out, cut_size_in]
    else:
        cut_size = [torch.tensor([degrees[subgraph].sum() - 2 * e_h \
                                  for e_h, subgraph in zip(e_h_s, subgraphs)], device=device)]
    return cut_size


def induced_subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False, num_nodes=None):
    # Returns the induced subgraph. 
    # Modified from pytorch geometric in order to return the label mapping as well
    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.ones(n_mask.size(0), dtype=torch.long,
                                device=device)*(-1)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
        else:
            n_idx = torch.arange(num_nodes, device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.ones(num_nodes, dtype=torch.long, device=device)*(-1)
            n_idx[subset] = torch.arange(subset.size(0), device=device)
        else:
            n_idx = torch.arange(num_nodes, device=device)

    if edge_index.numel()!=0:
        mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None

        if relabel_nodes:
            edge_index = n_idx[edge_index]

    return edge_index, edge_attr, n_idx


def remove_overlapping_subgraphs(graph,
                                 new_graph,
                                 remaining_nodes,
                                 num_nodes,
                                 relabelling,
                                 dictionary_size,
                                 relabel_nodes=True):
    device = graph.edge_index.device

    if hasattr(graph, 'subgraph_detections'):
        # import pdb;pdb.set_trace()
        graphs_subgraph_detections = []
        for graph_subgraph_detections in graph.subgraph_detections:
            deleted = torch.ones((num_nodes,), device=device)
            deleted[remaining_nodes] = 0
            detect_deleted = [
                deleted[subgraph_detections_atom] if subgraph_detections_atom.numel() != 0 else torch.tensor([]) \
                for subgraph_detections_atom in graph_subgraph_detections]

            # reject the subgraphs that cannot be selected anymore (remember the constraint graph)
            subgraph_detections = [subgraph_detections_atom[(detect_deleted_atom == 0).all(1)] \
                                   for (subgraph_detections_atom, detect_deleted_atom) \
                                   in zip(graph_subgraph_detections, detect_deleted) if
                                   subgraph_detections_atom.numel() != 0]

            # then relabel subgraphs
            subgraph_detections = [relabelling[subgraph_detections_atom] \
                                   for subgraph_detections_atom in subgraph_detections if
                                   subgraph_detections_atom.numel() != 0]

            graphs_subgraph_detections.append(subgraph_detections)

        setattr(new_graph, 'subgraph_detections', graphs_subgraph_detections)

    else:
        for i in range(dictionary_size):
            subgraph_detections_atom = getattr(graph, 'subgraph_index_' + str(i))
            if subgraph_detections_atom.numel() != 0:
                deleted = torch.ones((num_nodes,), device=device)
                deleted[remaining_nodes] = 0
                detect_deleted_atom = deleted[subgraph_detections_atom]
                remaining_subgraphs = (detect_deleted_atom == 0).all(0)

                # reject the subgraphs that cannot be selected anymore (remember the constraint graph)
                subgraph_detections_atom = subgraph_detections_atom[:, remaining_subgraphs]
                if relabel_nodes:
                    # then relabel subgraphs
                    subgraph_detections_atom = relabelling[subgraph_detections_atom]
                setattr(new_graph, 'subgraph_index_' + str(i), subgraph_detections_atom)

                subgraph_batch_index_atom = getattr(graph, 'subgraph_index_' + str(i) + '_batch')
                subgraph_batch_index_atom = subgraph_batch_index_atom[remaining_subgraphs]
                setattr(new_graph, 'subgraph_index_' + str(i) + '_batch', subgraph_batch_index_atom)

                if hasattr(graph, 'subgraph_labels_' + str(i)):
                    subgraph_labels_atom = getattr(graph, 'subgraph_labels_' + str(i))
                    setattr(new_graph, 'subgraph_labels_' + str(i), subgraph_labels_atom[remaining_subgraphs])
            else:
                setattr(new_graph, 'subgraph_index_' + str(i), subgraph_detections_atom)
                setattr(new_graph, 'subgraph_index_' + str(i) + '_batch', subgraph_detections_atom)
                if hasattr(graph, 'subgraph_labels_' + str(i)):
                   setattr(new_graph, 'subgraph_labels_' + str(i), subgraph_detections_atom)

    return new_graph