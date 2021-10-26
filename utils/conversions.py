import numpy as np
import graph_tool as gt
import graph_tool.stats as gt_stats
import networkx as nx
from scipy.sparse import csr_matrix

def convert_to_gt(edge_lists, directed=False):
    H_set = []
    for edge_list in edge_lists:
        H = gt.Graph(directed=directed)
        if np.array(edge_list).size == 0:
            H.add_vertex(1)
        else:
            H.add_edge_list(edge_list)
        gt_stats.remove_self_loops(H)
        gt_stats.remove_parallel_edges(H)
        H_set.append(H)

    return H_set


def convert_to_nx(E, n, directed=False, node_attrs=None, edge_attrs=None, node_attr_dims=None, edge_attr_dims=None):
    G = nx.Graph() if not directed else nx.DiGraph()
    if node_attr_dims is not None:
        G.add_nodes_from([(node, {str(i): node_attrs[node][i].item() for i in range(node_attr_dims)})
                          for node in range(int(n))])
    else:
        G.add_nodes_from(range(int(n)))
    if edge_attr_dims is not None:
        G.add_edges_from([(edge[0], edge[1], {str(i): edge_attrs[edge_enumer][i].item() for i in range(edge_attr_dims)})
                          for edge_enumer, edge in enumerate(E.transpose(1,0).tolist())])
    else:
        G.add_edges_from(E.transpose(1,0).tolist())
    return G


def convert_to_csr(E, n, directed=False, node_attrs=None, edge_attrs=None, node_attr_dims=None, edge_attr_dims=None):
    csr_graph = csr_matrix((np.ones((E.shape[1],)), (E[0].tolist(), E[1].tolist())), shape=(int(n), int(n)))
    #     csr_subgraph = to_scipy_sparse_matrix(E, edge_attr=None, num_nodes=n)
    return csr_graph

def convert_csr_to_nx(csr_graph, directed=False):
    G = nx.Graph() if not directed else nx.DiGraph()
    # if node_attr_dims is not None:
    #     G.add_nodes_from([(node, {str(i): node_attrs[node][i].item() for i in range(node_attr_dims)})
    #                       for node in range(int(n))])
    # else:
    # import pdb;pdb.set_trace()
    G.add_nodes_from(range(csr_graph.shape[0]))
    # if edge_attr_dims is not None:
    #     G.add_edges_from([(edge[0], edge[1], {str(i): edge_attrs[edge][i].item() for i in range(edge_attr_dims)})
    #                       for edge in E.transpose(1,0).tolist()])
    # else:
    G.add_edges_from(csr_graph.nonzero().tolist())
    return G