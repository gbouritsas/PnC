import torch
from torch.distributions.categorical import Categorical
from utils.utils_subgraphs import induced_subgraph, compute_cut_size_pairs, compute_combinations
from utils.visualisation import visualise_intermediate

class CompressionAgentNeuralPart:
    def __init__(self,
                 policy_network,
                 env,
                 train=True,
                 **kwargs):
        self.policy = policy_network
        self.env = env
        self.train = train
        self.attr_mapping = kwargs['attr_mapping']
        self.n_h_max = kwargs['n_h_max']
        self.n_h_min = kwargs['n_h_min']
        self.n_h_max_dict = kwargs['n_h_max_dict']
        self.n_h_min_dict = kwargs['n_h_min_dict']
        self.sampling_mechanism = kwargs['sampling_mechanism']
        # self.sample_k = kwargs['sample_k']
        # self.connected_sampling = kwargs['connected_sampling']

    def termination_criterion(self, graph_sizes):
        return (graph_sizes == 0).all()

    def compress(self, graph, **kwargs):
        device = graph.graph_size.device
        # visualisation arguments
        visualise = kwargs['visualise'] if 'visualise' in kwargs else False
        inds_to_visualise = kwargs['inds_to_visualise'] if 'inds_to_visualise' in kwargs else 0
        visualise_data_dict = {}
        # train vs inference mode: IMPORTANT
        if self.train:
            self.policy.train()
            self.query_inds = self.env.sorted_atoms_in_dict()
        else:
            self.policy.eval()
            self.query_inds = self.env.sorted_atoms_in_dict(kwargs['x_a'])
        # initialisation of parameters
        init_graph = graph
        node_labels_original = torch.arange(int(init_graph.graph_size.sum()), device=device)
        mask_t = torch.ones((init_graph.x.shape[0],), device=device).bool()
        n_h, e_h, atom_indices = [], [], []
        b = (-1) * torch.ones((len(init_graph.graph_size),), dtype=torch.long, device=device)
        subgraph_nodes_t, subgraph_nodes_all, log_probs = [], [], []
        # check termination criterion
        termination = self.termination_criterion(init_graph.graph_size)
        # get initial GNN predictions
        h_0_nodes, h_0_graph = self.policy(init_graph)
        global_context = None
        # ----------- iterative subgraph decomposition
        t = 0
        while not termination:
            # update context vectors
            global_context, global_context_updated = self.policy.aggregate_context(init_graph,
                                                                                   h_0_nodes,
                                                                                   subgraph_nodes_t,
                                                                                   global_context)
            h_0_nodes_restricted = h_0_nodes[node_labels_original]
            subgraphs_t, log_probs_t = self.select_subgraph(graph,
                                                            h_0_nodes_restricted,
                                                            h_0_graph,
                                                            global_context_updated,
                                                            self.policy)
            n_h.append(subgraphs_t['n_h'])
            e_h.append(subgraphs_t['e_h'])
            atom_indices.append(subgraphs_t['atom_indices'])
            # map the nodes to their initial indices
            subgraphs_t_nodes_relabelled = [node_labels_original[subgraphs_t['nodes'][i]]
                                            for i in range(len(subgraphs_t['nodes']))]
            subgraph_nodes_all.append(subgraphs_t_nodes_relabelled)
            subgraph_nodes_t = torch.cat(subgraphs_t_nodes_relabelled)
            log_probs.append(log_probs_t)
            # visualise
            if visualise:
                # TEST THAT
                visualise_data_dict = visualise_intermediate(visualise_data_dict,
                                                             inds_to_visualise,
                                                             init_graph,
                                                             mask_t,
                                                             subgraphs_t_nodes_relabelled,
                                                             subgraphs_t['atom_indices'],
                                                             attr_mapping=self.attr_mapping,
                                                             node_attr_dims=self.env.isomorphism_module.node_attr_dims,
                                                             edge_attr_dims=self.env.isomorphism_module.edge_attr_dims)
                new_mask = torch.ones_like(mask_t)
                new_mask[subgraph_nodes_t] = 0
                mask_t = new_mask & mask_t
            # remove the selected subgraph to reduce memory footprint and speed-up computations
            new_graph, remaining_nodes, _ = self.env.step(graph,
                                                          subgraphs_t['nodes'],
                                                          subgraphs_t['n_h'],
                                                          subgraphs_t['e_h'],
                                                          c_s=None,
                                                          relabel_nodes=True,
                                                          candidate_subgraphs=False)

            # check termination criterion
            b[(new_graph.graph_size == 0) & (b == -1)] = t + 1
            graph = new_graph
            termination = self.termination_criterion(graph.graph_size)
            # keep track of the initial indices
            node_labels_original = node_labels_original[remaining_nodes]
            if termination:
                break
            else:
                t += 1
        n_h = torch.stack(n_h).transpose(1, 0)
        e_h = torch.stack(e_h).transpose(1, 0)
        atom_indices = torch.stack(atom_indices).transpose(1, 0)
        cut_matrices = []
        subgraph_nodes_all = list(map(list, zip(*subgraph_nodes_all)))
        for i in range(len(init_graph.graph_size)):
            cut_matrices.append(compute_cut_size_pairs(init_graph,
                                                       subgraph_nodes_all[i],
                                                       directed=self.env.directed))
        cut_matrices = torch.stack(cut_matrices)
        c_ij = cut_matrices[torch.triu(torch.ones_like(cut_matrices), diagonal=1) == 1].view(cut_matrices.shape[0], -1) \
            if not self.env.directed else cut_matrices.view(cut_matrices.shape[0], -1)
        n_h_ij = compute_combinations(n_h, directed=self.env.directed)
        e_h_ij = compute_combinations(e_h, directed=self.env.directed)
        subgraphs = {'n_0': init_graph.graph_size,
                     'e_0': init_graph.edge_size,
                     'b': b,
                     'n_h': n_h,
                     'e_h': e_h,
                     'atom_indices': atom_indices,
                     'c_ij': c_ij,
                     'n_h_ij': n_h_ij,
                     'e_h_ij': e_h_ij}
        log_probs = torch.stack(log_probs).transpose(1, 0)
        return subgraphs, log_probs, visualise_data_dict



    def select_subgraph(self, graph, h_0_nodes, h_0_graph, global_context_updated, policy):
        graph_edge_features = graph.edge_features if hasattr(graph, 'edge_features') else None
        n_h = torch.zeros_like(graph.graph_size)
        e_h = torch.zeros_like(graph.graph_size)
        atom_indices = -1 * torch.ones_like(graph.graph_size).long()
        subgraph_nodes = []
        log_probs = torch.zeros_like(graph.graph_size)
        nodes_all = torch.arange(graph.x.shape[0], device=graph.x.device)
        # ------- sample one subgraph for each graph
        for i in range(len(graph.graph_size)):
            nodes_i = nodes_all[graph.batch == i]
            num_nodes_i = len(nodes_i)
            if num_nodes_i == 0:
                subgraph_nodes.append(nodes_i)
                continue
            if self.sampling_mechanism == 'sampling_without_replacement':
                subgraph_nodes_i, log_prob_nodes = compute_ordered_set_probs(h_0_nodes,
                                                                             h_0_graph[i:i+1],
                                                                             global_context_updated[i:i+1],
                                                                             policy,
                                                                             self.n_h_min,
                                                                             nodes_i,
                                                                             graph)
                log_probs[i] = log_prob_nodes
                subgraph_nodes_i = subgraph_nodes_i.view(-1)
            else:
                raise NotImplementedError(
                    "Sampling mechanism {} is not currently supported.".format(self.sampling_mechanism))
            # edges
            subgraph_edges_i, subgraph_edge_features_i, relabelling_i = induced_subgraph(subgraph_nodes_i,
                                                                                         graph.edge_index,
                                                                                         edge_attr=graph_edge_features,
                                                                                         relabel_nodes=True,
                                                                                         num_nodes=int(graph.graph_size.sum()))
            subgraph_nodes.append(subgraph_nodes_i)
            n_h[i] = len(subgraph_nodes_i)
            e_h[i] = subgraph_edges_i.shape[1] / 2 if not self.env.directed else subgraph_edges_i.shape[1]
            if n_h[i] <= self.n_h_max_dict and n_h[i] >= self.n_h_min_dict:
                atom_indices[i], G_i = self.env.map_to_dict(n_h[i], e_h[i], subgraph_edges_i,
                                                            self.env.dictionary_num_vertices,
                                                            self.env.dictionary_num_edges, self.env.dictionary,
                                                            self.query_inds, self.env.directed,
                                                            self.attr_mapping, graph.x[subgraph_nodes_i],
                                                            subgraph_edge_features_i)
                if atom_indices[i] == -1:
                    update_condition = self.train and \
                                       self.env.universe_type == 'adaptive' \
                                       and len(self.env.dictionary) < self.env.max_dict_size
                    if update_condition:
                        atom_indices[i] = len(self.env.dictionary)
                        # update the current version of the dictionary #
                        self.query_inds.append(len(self.env.dictionary))
                        self.env.update_dict_atoms([G_i],
                                                   [n_h[i]],
                                                   [e_h[i]])
        selected_subgraphs = {'n_h': n_h,
                              'e_h': e_h,
                              'atom_indices': atom_indices,
                              'nodes': subgraph_nodes}

        return selected_subgraphs, log_probs

def compute_ordered_set_probs(h_0_nodes, h_0_graph,
                              global_context_updated,
                              policy,
                              k_min,
                              candidate_nodes, graph):
    device = h_0_graph.device
    edge_index = graph.edge_index
    subgraph_nodes, log_prob = [], 0
    candidate_nodes, candidate_nodes_set = candidate_nodes.tolist(), set()
    num_nodes = len(candidate_nodes)
    if num_nodes <= k_min:
        selected_k = torch.tensor(num_nodes, device=device)
    else:
        graph_logit = policy.predict_graph(h_0_graph, global_context_updated)
        k_distribution = Categorical(logits=graph_logit.view(-1)[0:num_nodes - k_min + 1])
        selected_k = k_distribution.sample() + k_min
    batch = torch.zeros((len(candidate_nodes),), dtype=torch.long, device=device)
    init_node_logits = torch.zeros((len(h_0_nodes), 1), device=device)
    init_node_logits[candidate_nodes] = policy.predict_node(h_0_nodes[candidate_nodes], global_context_updated, batch)
    while len(candidate_nodes) != 0:
        if len(subgraph_nodes) == selected_k:
            break
        curr_logits = init_node_logits[candidate_nodes].view(-1)
        node_distribution = Categorical(logits=curr_logits)
        selected_ind = node_distribution.sample()
        log_prob += node_distribution.log_prob(selected_ind)
        new_node = candidate_nodes[selected_ind]
        subgraph_nodes.append(new_node)
        new_node_nbs = edge_index[1, edge_index[0] == new_node]
        candidate_nodes_set.update(new_node_nbs.tolist())
        candidate_nodes_set.difference_update(subgraph_nodes)
        candidate_nodes = list(candidate_nodes_set)
    subgraph_nodes = torch.tensor(subgraph_nodes, device=device)
    if num_nodes > k_min:
        log_prob += k_distribution.log_prob(selected_k - k_min)
    return subgraph_nodes, log_prob

