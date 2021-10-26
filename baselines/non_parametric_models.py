import sys
sys.path.append('../')
import argparse
import utils.parsing as parse
import os

import torch
import numpy as np
import random

import sknetwork.clustering as clustering
from sknetwork.utils import edgelist2adjacency
from utils.prepare_data import prepare_dataset, prepape_input_features, prepare_dataloaders
from utils.combinatorics import torch_log_factorial, torch_log_binom
from tqdm import tqdm
import graph_tool as gt
import graph_tool.stats as gt_stats
import graph_tool.inference as gt_inference
import math
import matplotlib.pyplot as plt
from encoding_decoding.encoding_costs import compute_cost_attr
plt.switch_backend('cairo')
epsilon = 1e-7
def node_length(n_max):
    return torch.log2(n_max + 1)

def edge_length(n):
    return torch.log2(n*(n-1)/2 + 1)

def edge_list_model_cost(n, e, n_max):
    L_n = node_length(n_max)
    L_e = edge_length(n)
    L_edges = e * torch.log2(n * (n - 1) / 2)
    return L_n + L_e + L_edges

def erdos_renyi_model_cost(n, e, n_max):
    L_n = node_length(n_max)
    L_e = edge_length(n)
    L_edges = torch_log_binom(n * (n - 1) / 2, e)
    return L_n + L_e + L_edges


def adjacency_cost(n, n_max):
    L_n = node_length(n_max)
    L_adjacency = n*(n-1)/2
    return L_n + L_adjacency

def main(args):
    ## ----------------------------------- infrastructure

    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args['np_seed'])
    os.environ['PYTHONHASHSEED'] = str(args['seed'])
    random.seed(args['seed'])
    print('[info] Setting all random seeds {}'.format(args['seed']))
    torch.set_num_threads(args['num_threads'])

    if args['wandb']:
        print('Wandb logging activated... ')
        import wandb
        wandb.init(sync_tensorboard=False, project=args['wandb_project'],
                   reinit = False, config = args, entity=args['wandb_entity'])
        print('[info] Monitoring with wandb')

    path = os.path.join(args['root_folder'], args['dataset'], args['dataset_name'])

    ## ----------------------------------- preparation of model components

    # generate/load dataset + detect/load subgraphs
    graphs_ptg = prepare_dataset(path,
                                 args['dataset'],
                                 args['dataset_name'],
                                 directed=args['directed'],
                                 H_set=[],
                                 multiprocessing=None,
                                 num_processes=None,
                                 candidate_subgraphs=False)
    # one-hot encoding etc of input features
    graphs_ptg, in_features_dims_dict, attr_mapping = prepape_input_features(args, graphs_ptg, path)
    print('Num graphs: {}'.format(len(graphs_ptg)))
    n_max = torch.tensor(max([graph.graph_size for graph in graphs_ptg]))

    if args['node_attr_encoding'] is not None:
        node_attr_unique_values = torch.tensor(in_features_dims_dict['node_attr_unique_values'])
    if args['edge_attr_encoding'] is not None:
        edge_attr_unique_values = torch.tensor(in_features_dims_dict['edge_attr_unique_values'])

    fold_idxs = args['fold_idx']
    for fold_idx in fold_idxs:
        print('############# FOLD NUMBER {:01d} #############'.format(fold_idx))
        # prepare dataloaders
        loader_train, loader_test, loader_val = prepare_dataloaders(args,
                                                                    graphs_ptg,
                                                                    path,
                                                                    fold_idx,
                                                                    candidate_subgraphs=False)
        loaders = [loader_train]
        dataset_names = ['train']
        if loader_test is not None:
            loaders += [loader_test]
            dataset_names += ['test']
        if loader_val is not None:
            loaders += [loader_val]
            dataset_names += ['val']

        cost_null_all = []
        num_edges_all = []
        log, log_args = '', []
        for i, loader in enumerate(loaders):
            cost_null = []
            num_edges = 0
            dataset_name = dataset_names[i]
            for iter, data in enumerate(tqdm(loader)):
                n = data.graph_size
                e = data.edge_size
                if args['non_parametric_model'] in ['sbm', 'Louvain', 'PropagationClustering']:
                    G_gt = gt.Graph(directed=args['directed'])
                    G_gt.add_edge_list(list(data.edge_index.transpose(1, 0).cpu().numpy()))
                    gt_stats.remove_self_loops(G_gt)
                    gt_stats.remove_parallel_edges(G_gt)
                    if args['non_parametric_model'] == 'sbm':
                        block_state = gt_inference.minimize_blockmodel_dl(G_gt,
                                                                          # deg_corr=False,
                                                                          state_args={'deg_corr':False},
                                                                          # mcmc_args=
                                                                          multilevel_mcmc_args={'entropy_args':{
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
                    else:
                        edge_list = data.edge_index.transpose(1, 0).cpu().numpy()
                        adjacency = edgelist2adjacency(edge_list)
                        cluster_labels = getattr(clustering, args['non_parametric_model'])().fit_transform(adjacency)
                        bs = G_gt.new_vertex_property('int')
                        bs.a = cluster_labels
                        block_state = gt_inference.BlockState(G_gt, bs, deg_corr=False)
                    entropy = block_state.entropy(dl=True,
                                                 partition_dl=True,
                                                 degree_dl=False,
                                                 edges_dl=True,
                                                 dense=True,
                                                 multigraph=False,
                                                  deg_entropy=False,
                                                  recs=False,
                                                  recs_dl=False,
                                                  beta_dl=1.0)/math.log(2.0)
                    # !! In graph-tool the description length accounts for the ordering of the nodes.
                    # Here, we modified the dl computation in order to drop the terms that recover the ordering.
                    # The change is done to the inference/blockmodel.py file (see Readme).
                    # This allows graph-tool to also optimise w.r.t the unlabelled version of the dl.
                    # In case, one cares only about computation and not optimisation, the change can be alternatively done below
                    if not args['modified_gt_entropy']:
                        n_h = torch.tensor(block_state.get_nr().a)
                        entropy -= torch_log_factorial(n) - torch_log_factorial(n_h).sum()
                    entropy = entropy + node_length(n_max) + edge_length(n)
                    if args['visualise'] and iter in args['inds_to_visualise'] \
                            and args['wandb'] and dataset_name=='test':
                        fig = plt.figure(figsize=(18,18))
                        fig.suptitle('num clusters {}'.format(block_state.B))
                        block_state.draw(vertex_size=0.1, edge_pen_width=0.02, mplfig=fig)
                        wandb.log({"clustering {}".format(iter): wandb.Image(fig)}, step=iter)
                        plt.close()
                        # state.draw(output="blocks_mdl.svg")
                    cost_null.append(torch.tensor([entropy]))
                elif args['non_parametric_model'] == 'edge_list':
                    cost_null.append(edge_list_model_cost(n, e, n_max))
                elif args['non_parametric_model'] == 'erdos_renyi':
                    cost_null.append(erdos_renyi_model_cost(n, e, n_max))
                elif args['non_parametric_model'] == 'uniform':
                    cost_null.append(adjacency_cost(n, n_max))
                else:
                    raise NotImplementedError
                if args['node_attr_encoding'] is not None:
                    cost_null[-1] += compute_cost_attr(args['node_attr_encoding'], n, node_attr_unique_values)
                if args['edge_attr_encoding'] is not None:
                    cost_null[-1] += compute_cost_attr(args['edge_attr_encoding'], e, edge_attr_unique_values)

                num_edges += data.edge_size.sum()
            cost_null_all.append(torch.cat(cost_null).sum())
            num_edges_all.append(num_edges)
            log += ',null {} bpe : {:.2f} bits : {:.2f}'
            log_args += [dataset_name, cost_null_all[i] / num_edges_all[i], cost_null_all[i]]
            if args['wandb']:
                wandb.run.summary['null_' + dataset_name + '_bpe_fold_' + str(fold_idx)] = (cost_null_all[i] / num_edges_all[i]).item()
                wandb.run.summary['null_' + dataset_name + '_bits_fold_' + str(fold_idx)] = cost_null_all[i].item()

        log += ',null total bpe : {:.2f} bits : {:.2f}'
        log_args += [sum(cost_null_all)/sum(num_edges_all), sum(cost_null_all)]
        if args['wandb']:
            wandb.run.summary['null_total_bpe_fold_' + str(fold_idx)] = (sum(cost_null_all)/sum(num_edges_all)).item()
            wandb.run.summary['null_total_bits_fold_' + str(fold_idx)] = sum(cost_null_all).item()

        print(log.format(*log_args))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #----------------- seeds
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=10)
    #----------------- experiment logs
    parser.add_argument('--wandb', type=parse.str2bool, default=False)
    parser.add_argument('--wandb_project', type=str, default="graph_compression")
    parser.add_argument('--wandb_entity', type=str, default="epfl")
    parser.add_argument('--results_folder', type=str, default='non_parametric_experiments')
    parser.add_argument('--visualise', type=parse.str2bool, default=False)
    parser.add_argument('--inds_to_visualise', type=parse.str2list2int, default=[0])
    #----------------- dataset and split
    parser.add_argument('--root_folder', type=str, default='../datasets/')
    parser.add_argument('--dataset', type=str, default='bioinformatics')
    parser.add_argument('--dataset_name', type=str, default='MUTAG')
    parser.add_argument('--directed', type=parse.str2bool, default=False)
    parser.add_argument('--fold_idx', type=parse.str2list2int, default=[0])
    parser.add_argument('--split', type=str, default='given')
    parser.add_argument('--split_seed', type=int, default=0) # only for random splitting
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=parse.str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    # ----------------- node and edge attribute encoding
    parser.add_argument('--node_attr_encoding', type=str, default=None) #'uniform'
    parser.add_argument('--edge_attr_encoding', type=str, default=None) #'uniform'

    parser.add_argument('--non_parametric_model', type=str, default='erdos_renyi')
    parser.add_argument('--modified_gt_entropy', type=parse.str2bool, default=False)

    args = parser.parse_args()
    print(args)
    main(vars(args))