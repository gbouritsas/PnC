import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('cairo')
import math
import wandb
from utils.conversions import convert_csr_to_nx
import graph_tool as gt
import graph_tool.stats as gt_stats
import graph_tool.inference as gt_inference

from matplotlib import cm
colormap =  cm.get_cmap('Set1')

def visualise_intermediate(visualise_data_dict,
                           inds_to_visualise,
                           graph,
                           mask,
                           subgraphs,
                           atom_indices=None,
                           attr_mapping=None,
                           node_attr_dims=None,
                           edge_attr_dims=None):


    for ind_to_visualise in inds_to_visualise:
        n_mask = (graph.batch==ind_to_visualise) & mask
        if n_mask.sum()==0:
            continue
        edge_mask = n_mask[graph.edge_index[0]] & n_mask[graph.edge_index[1]]

        edges_vis = graph.edge_index[:, edge_mask]
        nodes_vis = torch.where(n_mask)[0]
        G_vis = nx.Graph()
        node_label_list = nodes_vis.tolist()
        edge_label_list = edges_vis.transpose(1,0).tolist()
        G_vis.add_nodes_from(node_label_list)
        G_vis.add_edges_from(edge_label_list)
        if node_attr_dims is not None or edge_attr_dims is not None:
            node_attrs, edge_attrs = attr_mapping.map(graph.x[n_mask],
                                                      graph.edge_features[edge_mask]
                                                      if hasattr(graph, 'edge_features') else None)
        if node_attr_dims is not None:
            node_labels = {node_label: ','.join([attr_mapping.node_attr_values[i][node_attrs[node_ind,i].item()]
                                           for i in range(node_attr_dims)])
                           for node_ind, node_label in enumerate(node_label_list)}
        else:
            node_labels = None
        if edge_attr_dims is not None:
            edge_labels = {(edge_label[0], edge_label[1]): ','.join([attr_mapping.edge_attr_values[i][edge_attrs[edge_ind,i].item()]
                                           for i in range(edge_attr_dims)])
                           for edge_ind, edge_label in enumerate(edge_label_list)}
        else:
            edge_labels = None
        node_color = ['b' if i not in subgraphs[ind_to_visualise] \
                      else 'r' for i in list(G_vis.nodes())]
        edge_color = ['k' if e[0] not in subgraphs[ind_to_visualise] \
                      or e[1] not in subgraphs[ind_to_visualise] \
                      else 'r' for e in list(G_vis.edges())]
        edge_width = [1 if e[0] not in subgraphs[ind_to_visualise] \
                      or e[1] not in subgraphs[ind_to_visualise] \
                      else 2 for e in list(G_vis.edges())]
        curr_visualisation = {'G': G_vis,
                              'cluster_labels': subgraphs[ind_to_visualise],
                              'node_color': node_color,
                              'edge_color': edge_color,
                              'edge_width': edge_width,
                              'node_labels': node_labels,
                              'edge_labels': edge_labels,
                              'atom_index': atom_indices[ind_to_visualise]}
        if ind_to_visualise not in visualise_data_dict:
            visualise_data_dict[ind_to_visualise] = [curr_visualisation]
        else:
            visualise_data_dict[ind_to_visualise].append(curr_visualisation)
    return visualise_data_dict

def visualise_partitioning(G, cluster_labels_list, mv_fractional, fig, directed=False, hs_scaling=3):
    mapping = {node:i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    G_gt = gt.Graph(directed=directed)
    G_gt.add_vertex(max(G.nodes())+1)
    G_gt.add_edge_list(list(G.edges()))
    gt_stats.remove_self_loops(G_gt)
    gt_stats.remove_parallel_edges(G_gt)
    bs = G_gt.new_vertex_property('int')
    bs.a = np.ones((G.number_of_nodes(),)) * len(cluster_labels_list)
    hs = G_gt.new_vertex_property('double')
    hs.a = 1 + 3 * np.ones_like(bs.a)
    for cluster, cluster_labels in enumerate(cluster_labels_list):
        cluster_labels = [mapping[cluster_label] for cluster_label in cluster_labels]
        bs.a[cluster_labels] = cluster
        hs.a[cluster_labels] = 1 + hs_scaling * (1 - mv_fractional[cluster])
    block_state = gt_inference.BlockState(G_gt, bs)
    block_state.draw(vertex_halo=True, vertex_halo_size=hs, vertex_size=0.2, edge_pen_width=0.05, mplfig=fig)

    return block_state



def visualisation_log(visualise_data_dict,
                      inds_to_visualise,
                      visualise_step,
                      total_costs=None,
                      baseline=None,
                      cost_terms=None,
                      x_a_fractional=None,
                      partioning=True,
                      directed=False,
                      subgraphs=None,
                      graphs=None):

    if visualise_data_dict is None and subgraphs is not None:
        for ind_to_visualise in inds_to_visualise:
            fig = plt.figure()
            cluster_labels_list = [cluster_labels.tolist() for cluster_labels in subgraphs[ind_to_visualise]['cluster_labels']]
            mv_fractional = x_a_fractional[subgraphs[ind_to_visualise]['atom_indices'][0]].tolist()
            mv_fractional[mv_fractional==-1] = 0
            G_vis = nx.Graph()
            G_vis.add_nodes_from(list(range(int(graphs[ind_to_visualise].graph_size))))
            G_vis.add_edges_from(graphs[ind_to_visualise].edge_index.transpose(1,0).tolist())
            block_state = visualise_partitioning(G_vis,
                                                 cluster_labels_list,
                                                 mv_fractional,
                                                 fig,
                                                 directed=directed)
            baseline_dl = baseline[ind_to_visualise] if baseline is not None else 0
            dl = total_costs[ind_to_visualise]
            if cost_terms is None:
                fig.suptitle('num clusters: {}, baseline dl: {:.2f}, dl: {:.2f}'.
                                    format(max(block_state.get_blocks()) + 1, baseline_dl, dl))
            else:
                print_log = 'num clusters: {}, baseline dl: {:.2f}, dl: {:.2f}, b: {:.2f}, H_dict: {:.2f}, H_null: {:.2f}, C: {:.2f}, '
                print_data = [max(block_state.get_blocks()) + 1, baseline_dl, dl] \
                             + visualise_data_dict[ind_to_visualise][0]['cost_terms']
                fig.suptitle(print_log.format(*print_data))


            wandb.log({'graph_partitioning_'+str(ind_to_visualise): wandb.Image(plt)}, step=visualise_step)
            plt.close()
    else:
        for ind_to_visualise in inds_to_visualise:
            if ind_to_visualise in visualise_data_dict:
                if baseline is not None:
                    visualise_data_dict[ind_to_visualise][0]['dl'] = (baseline[ind_to_visualise].item(), total_costs[ind_to_visualise].item())
                else:
                    visualise_data_dict[ind_to_visualise][0]['dl'] = (0, total_costs[ind_to_visualise].item())
                if cost_terms is not None:
                    visualise_data_dict[ind_to_visualise][0]['cost_terms'] = [cost_terms['b'][ind_to_visualise].item(),
                                                                             cost_terms['H_dict'][ind_to_visualise].item(),
                                                                             cost_terms['H_null'][ind_to_visualise].item(),
                                                                             cost_terms['C'][ind_to_visualise].item()]


                visualise_data_list = visualise_data_dict[ind_to_visualise]
                num_rows = int(math.ceil((len(visualise_data_list)+1)/ 2)) if partioning else int(math.ceil(len(visualise_data_list) / 2))
                fig, ax = plt.subplots(num_rows, 2, figsize=(18,18))
                if x_a_fractional is not None:
                    mv_fractional = [x_a_fractional[visualise_data['atom_index']].item()
                                     if visualise_data['atom_index'] != -1 else 0 for visualise_data in
                                     visualise_data_list]
                else:
                    mv_fractional = [0 for _ in visualise_data_list]
                for i,visualise_data in enumerate(visualise_data_list):
                    ax_i = ax[i//2,i%2] if num_rows >= 2 else ax[i%2]
                    pos = nx.spring_layout(visualise_data['G'], scale=2)
                    if visualise_data['edge_labels'] is not None:
                        nx.draw_networkx_edge_labels(visualise_data['G'],
                                                     pos=pos,
                                                     edge_labels=visualise_data['edge_labels'],
                                                     ax=ax_i)
                    nx.draw(visualise_data['G'],
                            pos=pos,
                            node_color=visualise_data['node_color'],
                            edge_color=visualise_data['edge_color'],
                            width=visualise_data['edge_width'],
                            with_labels=True,
                            labels=visualise_data['node_labels'],
                            ax=ax_i)
                    if 'dl' in visualise_data:
                        print_value = [visualise_data['dl'][0], visualise_data['dl'][1],
                                       visualise_data['atom_index'].item(),
                                       mv_fractional[i]]
                    else:
                        print_value = [0, 0,
                                       visualise_data['atom_index'].item(),
                                       mv_fractional[i]]
                    ax_i.title.set_text('Baseline DL: {:.4f}, DL: {:.4f},'
                                        ' atom_index: {},  mv: {}'.format(*print_value))

                if partioning:
                    ax_i = ax[len(visualise_data_list) // 2, len(visualise_data_list) % 2] if num_rows >= 2\
                        else ax[len(visualise_data_list) % 2]
                    cluster_labels_list = [visualise_data['cluster_labels'].tolist() for visualise_data in visualise_data_list]
                    block_state = visualise_partitioning(visualise_data_list[0]['G'],
                                                             cluster_labels_list,
                                                             mv_fractional,
                                                             ax_i,
                                                             directed=directed)
                    if 'dl' in visualise_data_list[0]:
                        baseline_dl = visualise_data_list[0]['dl'][0]
                        dl = visualise_data_list[0]['dl'][1]
                        if cost_terms is None:
                            ax_i.title.set_text('num clusters: {:.4f}, baseline dl: {:.4f}, dl: {:.4f}'.
                                                format(max(block_state.get_blocks()) + 1,baseline_dl, dl))
                        else:
                            print_log = 'num clusters: {}, baseline dl: {:.2f}, dl: {:.2f}, b: {:.2f}, H_dict: {:.2f}, H_null: {:.2f}, C: {:.2f}, '
                            print_data = [max(block_state.get_blocks()) + 1, baseline_dl, dl] \
                                         + visualise_data_dict[ind_to_visualise][0]['cost_terms']
                            ax_i.title.set_text(print_log.format(*print_data))
                    else:
                        ax_i.title.set_text('num clusters: {}'.format(max(block_state.get_blocks()) + 1))
            wandb.log({'graph_'+str(ind_to_visualise): wandb.Image(plt)}, step=visualise_step)
            plt.close()

    return


def visualise_subgraphs(atoms, init_i, final_i,visualise_step=0, data_format='nx',
                        attr_mapping=None, node_attr_dims=None, edge_attr_dims=None, color_attrs=True):

    for i in range(init_i, final_i):
        # G_s = nx.Graph()
        # G_s.add_edges_from(edge_lists[i].cpu().numpy().transpose(1,0))
        # G_s.add_nodes_from(range(0,int(num_vertices[i])))
        fig = plt.figure(figsize=(8,8), dpi=300);
        if data_format == 'csr':
            G = convert_csr_to_nx(atoms[i])
        else:
            G = atoms[i]
        pos = nx.spring_layout(G, scale=2)
        node_colors = []
        if node_attr_dims is not None:
            if color_attrs:
                if len(attr_mapping.node_attr_values)>1:
                    print('fix this')
                    import pdb;pdb.set_trace()
                else:
                    pass
                    #groups = set(attr_mapping.node_attr_values[0])
                    #mapping = dict(zip(count(), sorted(groups)))
            node_labels = {}
            for node in (G.nodes(data=True)):
                node_ind = node[0]
                attr_dict = node[1]
                node_labels[node_ind] = ','.join([attr_mapping.node_attr_values[int(k)][v] for k,v in attr_dict.items()])
                node_colors.append(colormap(attr_dict['0']/len(attr_mapping.node_attr_values[0])))
        else:
            node_labels = None
        if edge_attr_dims is not None:
            edge_labels = {}
            for edge in (G.edges(data=True)):
                edge_ind = (edge[0], edge[1])
                attr_dict = edge[2]
                edge_labels[edge_ind] = ','.join([attr_mapping.edge_attr_values[int(k)][v] for k,v in attr_dict.items()])
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=25)
        if len(node_colors)==0:
            nx.draw(G, pos=pos, with_labels=False, labels=node_labels)
        else:
            nx.draw(G, pos, nodelist=G.nodes(),
                    with_labels=True, labels=node_labels,
                    node_color=node_colors, node_size=1000, width=4.0, font_size=25)
            #plt.colorbar(nc)
            #plt.axis('off')
        # nx.draw(G, with_labels=True);
        wandb.log({'subgraph_'+str(i): wandb.Image(plt)}, step=visualise_step)
        #plt.savefig('./images/MUTAG/subgraph_'+str(i)+'.svg')
        #plt.savefig('./images/MUTAG/subgraph_' + str(i) + '.svg')
        plt.close();

    return