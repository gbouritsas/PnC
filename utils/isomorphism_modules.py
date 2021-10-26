from sknetwork.topology import are_isomorphic
import networkx as nx
import networkx.algorithms.isomorphism as iso
from utils.conversions import convert_to_csr, convert_to_nx
import timeout_decorator

class WLIsomoprhism:
    def __init__(self, node_attr_dims=None, edge_attr_dims=None):
        self.node_attr_dims = node_attr_dims
        self.edge_attr_dims = edge_attr_dims
        self.convert_graph = convert_to_csr
    def match(self, G1, G2):
        return are_isomorphic(G1, G2)

class ExactIsomoprhism:

    def __init__(self, node_attr_dims=None, edge_attr_dims=None):
        self.node_attr_dims = node_attr_dims
        self.edge_attr_dims = edge_attr_dims
        self.convert_graph = convert_to_nx
        self.nm = None if node_attr_dims is None else iso.categorical_node_match([str(i) for i in range(node_attr_dims)],
                                                                                 [0 for _ in range(node_attr_dims)])
        self.em = None if edge_attr_dims is None else iso.categorical_edge_match([str(i) for i in range(edge_attr_dims)],
                                                                                 [0 for _ in range(edge_attr_dims)])

    @timeout_decorator.timeout(1)
    def match(self, G1, G2):
        return nx.is_isomorphic(G1, G2, node_match=self.nm, edge_match=self.em)


def prepare_isomorphism_module(isomorphism_type, node_attr_dims=None, edge_attr_dims=None):
    if isomorphism_type == 'WL':
        if node_attr_dims is not None or edge_attr_dims is not None:
            raise NotImplementedError('colored WL not implemented')
        isomorphism_module = WLIsomoprhism(node_attr_dims=node_attr_dims, edge_attr_dims=edge_attr_dims)
    else:
        isomorphism_module = ExactIsomoprhism(node_attr_dims=node_attr_dims, edge_attr_dims=edge_attr_dims)
    return isomorphism_module
