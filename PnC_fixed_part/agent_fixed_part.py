# %%
import sys
sys.path.append('../')
import sknetwork.clustering as clustering
from sknetwork.utils import edgelist2adjacency
import torch
from utils.utils_subgraphs import induced_subgraph, compute_cut_size_pairs, compute_combinations
from utils.visualisation import visualise_intermediate
import graph_tool as gt
import graph_tool.stats as gt_stats
import graph_tool.inference as gt_inference


class CompressionAgentFixedPart:

    def __init__(self,
                 env,
                 train=True,
                 partitioning_algorithm='Louvain',
                 **kwargs):

        self.env = env
        self.train = train
        self.attr_mapping = kwargs['attr_mapping']
        self.n_h_max = kwargs['n_h_max']
        self.n_h_min = kwargs['n_h_min']
        self.n_h_max_dict = kwargs['n_h_max_dict']
        self.n_h_min_dict = kwargs['n_h_min_dict']
        self.partitioning_algorithm = partitioning_algorithm

    def compress(self, graph, **kwargs):

        device = graph.graph_size.device
        # visualisation arguments
        visualise = kwargs['visualise'] if 'visualise' in kwargs else False
        inds_to_visualise = kwargs['inds_to_visualise'] if 'inds_to_visualise' in kwargs else 0
        visualise_data_dict = {}

        # train vs inference mode: IMPORTANT
        self.query_inds = self.env.sorted_atoms_in_dict()

        # initialisation of parameters
        init_graph = graph
        mask_t = torch.ones((init_graph.x.shape[0],), device=device).bool()
        n_h, e_h, atom_indices = [], [], []
        subgraph_nodes_t = []
        subgraph_nodes_all = []

        if self.partitioning_algorithm == 'sbm':
            G_gt = gt.Graph(directed=self.env.directed)
            G_gt.add_edge_list(list(graph.edge_index.transpose(1, 0).cpu().numpy()))
            gt_stats.remove_self_loops(G_gt)
            gt_stats.remove_parallel_edges(G_gt)
            state = gt_inference.minimize_blockmodel_dl(G_gt,
                                                              # deg_corr=False,
                                                              state_args={'deg_corr': False},
                                                              # mcmc_args=
                                                              multilevel_mcmc_args={'entropy_args': {
                                                                  'dl': True,
                                                                  'partition_dl': True,
                                                                  'degree_dl': False,
                                                                  'edges_dl': True,
                                                                  'dense': True,
                                                                  'multigraph': False,
                                                                  'deg_entropy': False,
                                                                  'recs': False,
                                                                  'recs_dl': False,
                                                                  'beta_dl': 1.0}})
            cluster_labels = torch.tensor(state.get_blocks().get_array(), device=device)
        else:
            adjacency = edgelist2adjacency(graph.edge_index.transpose(1, 0).cpu().numpy())
            cluster_labels = torch.tensor(getattr(clustering, self.partitioning_algorithm)()
                                          .fit_transform(adjacency), device=device)
        clusters = cluster_labels.unique()
        for cluster in clusters:
            subgraphs_t = self.map_subgraph(graph, cluster, cluster_labels)
            n_h.append(subgraphs_t['n_h'])
            e_h.append(subgraphs_t['e_h'])
            atom_indices.append(subgraphs_t['atom_indices'])
            subgraph_nodes_all.append(subgraphs_t['nodes'])
            if visualise:
                # TEST THAT
                visualise_data_dict = visualise_intermediate(visualise_data_dict,
                                                             inds_to_visualise,
                                                             init_graph,
                                                             mask_t,
                                                             subgraphs_t['nodes'],
                                                             subgraphs_t['atom_indices'],
                                                             attr_mapping=self.attr_mapping,
                                                             node_attr_dims=self.env.isomorphism_module.node_attr_dims,
                                                             edge_attr_dims=self.env.isomorphism_module.edge_attr_dims)
                new_mask = torch.ones_like(mask_t)
                new_mask[subgraph_nodes_t] = 0
                mask_t = new_mask & mask_t


        n_h = torch.stack(n_h).transpose(1, 0)
        e_h = torch.stack(e_h).transpose(1, 0)
        atom_indices = torch.stack(atom_indices).transpose(1, 0)

        cut_matrices = []
        subgraph_nodes_all = list(map(list, zip(*subgraph_nodes_all)))
        cut_matrices.append(compute_cut_size_pairs(init_graph,
                                                   subgraph_nodes_all[0],
                                                   directed=self.env.directed))
        cut_matrices = torch.stack(cut_matrices)
        c_ij = cut_matrices[torch.triu(torch.ones_like(cut_matrices), diagonal=1) == 1].view(cut_matrices.shape[0], -1) \
            if not self.env.directed else cut_matrices.view(cut_matrices.shape[0], -1)
        n_h_ij = compute_combinations(n_h, directed=self.env.directed)
        e_h_ij = compute_combinations(e_h, directed=self.env.directed)
        subgraphs = {'n_0': init_graph.graph_size,
                     'e_0': init_graph.edge_size,
                     'b':  len(clusters) * torch.ones((1,1), dtype=torch.long, device=device),
                     'n_h': n_h,
                     'e_h': e_h,
                     'atom_indices': atom_indices,
                     'c_ij': c_ij,
                     'n_h_ij': n_h_ij,
                     'e_h_ij': e_h_ij,
                     'cluster_labels': subgraph_nodes_all[0]}

        return subgraphs, visualise_data_dict

    def map_subgraph(self, graph, cluster, cluster_labels):

        graph_edge_features = graph.edge_features if hasattr(graph, 'edge_features') else None
        n_h = torch.zeros_like(graph.graph_size)
        e_h = torch.zeros_like(graph.graph_size)
        atom_indices = -1 * torch.ones_like(graph.graph_size).long()
        subgraph_nodes = []

        subgraph_nodes_i = torch.where(cluster_labels == cluster)[0]
        subgraph_edges_i, subgraph_edge_features_i, relabelling_i = induced_subgraph(subgraph_nodes_i,
                                                                                     graph.edge_index,
                                                                                     edge_attr=graph_edge_features,
                                                                                     relabel_nodes=True,
                                                                                     num_nodes=int(
                                                                                         graph.graph_size.sum()))
        subgraph_nodes.append(subgraph_nodes_i)
        n_h[0] = len(subgraph_nodes_i)
        e_h[0] = subgraph_edges_i.shape[1] / 2 if not self.env.directed else subgraph_edges_i.shape[1]
        if n_h[0] <= self.n_h_max_dict and n_h[0] >= self.n_h_min_dict:
            atom_indices[0], G_i = self.env.map_to_dict(n_h[0], e_h[0], subgraph_edges_i,
                                                        self.env.dictionary_num_vertices,
                                                        self.env.dictionary_num_edges, self.env.dictionary,
                                                        self.query_inds, self.env.directed,
                                                        self.attr_mapping, graph.x[subgraph_nodes_i],
                                                        subgraph_edge_features_i)
            if atom_indices[0] == -1:
                update_condition = self.train and \
                                   self.env.universe_type == 'adaptive' \
                                   and len(self.env.dictionary) < self.env.max_dict_size
                if update_condition:
                    atom_indices[0] = len(self.env.dictionary)
                    # update the current version of the dictionary #
                    self.query_inds.append(len(self.env.dictionary))
                    self.env.update_dict_atoms([G_i],
                                               [n_h[0]],
                                               [e_h[0]])
        selected_subgraphs = {'n_h': n_h,
                              'e_h': e_h,
                              'atom_indices': atom_indices,
                              'nodes': subgraph_nodes}
        return selected_subgraphs

