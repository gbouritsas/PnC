import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from copy import deepcopy
import json
import numpy as np
from utils.loss_evaluation_fns import *
from utils.visualisation import visualise_subgraphs, visualisation_log

def prepare_logs(args, path, fold_idx=''):
    # prepare result folder
    if 'model_name' in args:
        results_folder = os.path.join(path, 'results', args['results_folder'], str(fold_idx), args['model_name'])
    else:
        results_folder = os.path.join(path, 'results', args['results_folder'], str(fold_idx), args['partitioning_algorithm'])
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # prepare folder for model checkpoints
    checkpoint_path = os.path.join(results_folder, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # save parameters of the training job
    with open(os.path.join(results_folder, 'params.json'), 'w') as fp:
        saveparams = deepcopy(args)
        json.dump(saveparams, fp)

    return checkpoint_path


def test_neural_part(loader,
                     agent,
                     dictionary_probs_model,
                     evaluation_fn,
                     subgraphs=None,
                     visualise_data_dict=None,
                     n_iters_test=None,
                     **kwargs):
    device = next(agent.policy.parameters()).device

    subgraphs_all, atom_indices = [], []
    total_costs_all, baseline_costs_all = [], []
    empirical_subgraphs_types_freqs= [0, 0]
    empirical_b_freqs = torch.zeros((dictionary_probs_model.b_max,))
    total_num_edges = 0
    agent.train = False
    with torch.no_grad():
        x_a, b_probs, delta_prob, atom_probs, cut_size_probs = dictionary_probs_model.prune_universe(agent.train)
        kwargs_test = {'x_a': x_a, # map subgraphs only to the current dictionary
                       'visualise': kwargs['visualise'] if 'visualise' in kwargs else False,
                       'inds_to_visualise': kwargs['inds_to_visualise'] if 'inds_to_visualise' in kwargs else 0,
                       'visualise_step': kwargs['visualise_step'] if 'visualise_step' in kwargs else 0}
        kwargs_heuristic = {'compute_dl': agent.env.compute_dl,
                            'x_a': x_a,  # unused
                            'b_probs': b_probs,
                            'delta_prob': delta_prob,
                            'atom_probs': atom_probs}  # unused
        kwargs_test.update(kwargs_heuristic)

        data_iterator = iter(loader)
        n_iters = len(loader) if n_iters_test is None else n_iters_test
        for iteration in tqdm(range(n_iters)):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                data = next(data_iterator)
            data = data.to(device)
            kwargs_test['visualise'] = kwargs_test['visualise'] if iteration == 0 else False
            subgraphs, log_probs_actions, visualise_data_dict = agent.compress(data, **kwargs_test)
            total_costs, baseline, log_probs, cost_terms = agent.env.compute_dl(subgraphs,
                                                                                x_a,
                                                                                b_probs,
                                                                                delta_prob,
                                                                                atom_probs,
                                                                                cut_size_probs,
                                                                                log_probs_actions)
            total_costs_all.append(total_costs)
            baseline_costs_all.append(baseline)
            total_num_edges += data.edge_size.sum()
            subgraphs_all.append(subgraphs)
            empirical_b_freqs[subgraphs['b'] - 1] += 1
            atoms_in_dict = x_a[subgraphs['atom_indices']][subgraphs['n_h'] != 0].sum().item()
            empirical_subgraphs_types_freqs[0] += atoms_in_dict
            empirical_subgraphs_types_freqs[1] += (subgraphs['n_h'] != 0).sum().item() - atoms_in_dict
            atom_indices.append(subgraphs['atom_indices'])
            if kwargs_test['visualise']:
                visualisation_log(visualise_data_dict,
                                  kwargs_test['inds_to_visualise'],
                                  kwargs_test['visualise_step'],
                                  total_costs=total_costs,
                                  baseline=baseline,
                                  cost_terms=cost_terms,
                                  x_a_fractional=dictionary_probs_model.membership_vars(),
                                  partioning=True,
                                  directed=agent.env.directed,
                                  subgraphs=None,
                                  graphs=None)
    uncompressed_size = torch.cat(baseline_costs_all)
    compressed_size = torch.cat(total_costs_all)
    atom_costs = torch.zeros_like(x_a)
    atom_costs[0:len(agent.env.dictionary)] = agent.env.compute_atom_costs()
    dictionary_cost = (atom_costs *  x_a).sum()
    evaluation_metric_w_dict, evaluation_metric = evaluation_fn(compressed_size, uncompressed_size, dictionary_cost)
    atom_indices_flattened = torch.cat([atom_indices_batch.flatten() for atom_indices_batch in atom_indices])

    atom_indices_flattened = atom_indices_flattened[atom_indices_flattened != -1]
    if len(atom_indices_flattened) != 0:
        _, empirical_atom_probs = agent.env.estimate_atom_freqs_probs(atom_indices_flattened)
    else:
        empirical_atom_probs = torch.zeros_like(atom_probs)
    metrics_dict = {"compressed_size": compressed_size.sum().item(),
                    "uncompressed_size": uncompressed_size.sum().item(),
                    'total_num_edges': total_num_edges.item(),
                    "evaluation_metric_w_dict": evaluation_metric_w_dict,
                    "evaluation_metric":evaluation_metric,
                    "dictionary_cost": dictionary_cost.item(),
                    "x_a": x_a,
                    'b_probs': b_probs,
                    'empirical_b_freqs': empirical_b_freqs,
                    'delta_prob': delta_prob,
                    'empirical_subgraphs_types_freqs': empirical_subgraphs_types_freqs,
                    "atom_probs": atom_probs,
                    "empirical_atom_probs": empirical_atom_probs,
                    'cut_size_probs': cut_size_probs}
    return metrics_dict




def test_fixed_part(loader,
                    agent,
                    dictionary_probs_model,
                    evaluation_fn,
                    subgraphs,
                    visualise_data_dict=None,
                    n_iters_test=None,
                    **kwargs):

    agent.train = False
    total_num_edges = sum([subgraph['e_0'] for subgraph in subgraphs])
    kwargs_test = {'visualise': kwargs['visualise'] if 'visualise' in kwargs else False,
                        'inds_to_visualise': kwargs['inds_to_visualise'] if 'inds_to_visualise' in kwargs else 0,
                        'visualise_step': kwargs['visualise_step'] if 'visualise_step' in kwargs else 0}
    with torch.no_grad():
        x_a, b_probs, delta_prob, atom_probs, cut_size_probs = dictionary_probs_model.prune_universe(agent.train)
        compressed_size, uncompressed_size = ml_estimation_fixed_part(subgraphs, agent,
                                                                      x_a, b_probs, delta_prob, atom_probs, cut_size_probs)
        if kwargs_test['visualise']:
            visualisation_log(visualise_data_dict,
                              kwargs_test['inds_to_visualise'],
                              kwargs_test['visualise_step'],
                              total_costs=compressed_size,
                              baseline=uncompressed_size,
                              x_a_fractional=dictionary_probs_model.membership_vars(),
                              partioning=True,
                              directed=agent.env.directed,
                              subgraphs=subgraphs,
                              graphs=loader.dataset)
        atom_costs = torch.zeros_like(x_a)
        atom_costs[0:len(agent.env.dictionary)] = agent.env.compute_atom_costs()
        dictionary_cost = (atom_costs *  x_a).sum()
        evaluation_metric_w_dict, evaluation_metric = evaluation_fn(compressed_size, uncompressed_size, dictionary_cost)

    metrics_dict = {"compressed_size": compressed_size.sum().item(),
                    "uncompressed_size": uncompressed_size.sum().item(),
                    'total_num_edges': total_num_edges.item(),
                    "evaluation_metric_w_dict": evaluation_metric_w_dict,
                    "evaluation_metric":evaluation_metric,
                    "dictionary_cost": dictionary_cost.item(),
                    "x_a": x_a,
                    'b_probs': b_probs,
                    'delta_prob': delta_prob,
                    "atom_probs":atom_probs,
                    'cut_size_probs': cut_size_probs}
    return metrics_dict





def realtime_logger(loaders,
                    dataset_names,
                    agent,
                    dictionary_probs_model,
                    evaluation_fn,
                    epoch,
                    fold_idx,
                    wandb_realtime,
                    initial_dict_size,
                    test_fn=test_neural_part,
                    subgraphs_all=None,
                    visualise_data_dict=None,
                    n_iters_test=None,
                    **kwargs):

    log = ''
    log_args = []
    log_corpus = {}
    visualise_dataset_name = 'test' if 'test' in dataset_names else 'train'
    raw_bits_model = []
    total_num_edges = []
    dataset_compression = []
    for i, loader in enumerate(loaders):
        dataset_name = dataset_names[i]
        subgraphs = subgraphs_all[i] if subgraphs_all is not None else None
        kwargs_test = kwargs if dataset_name==visualise_dataset_name else {}
        compression_metrics = test_fn(loader,
                                      agent,
                                      dictionary_probs_model,
                                      evaluation_fn,
                                      subgraphs,
                                      visualise_data_dict,
                                      n_iters_test,
                                      **kwargs_test)
        raw_bits_model.append(compression_metrics["compressed_size"])
        total_num_edges.append(compression_metrics["total_num_edges"])
        num_params = dictionary_probs_model.count_params(compression_metrics['x_a'][0:len(agent.env.dictionary)])
        parameter_cost = num_params * kwargs['bits_per_parameter']
        bpe_dict = bits_per_edge(raw_bits_model[i], total_num_edges[i], compression_metrics["dictionary_cost"])
        bpe_dict_param = bits_per_edge(raw_bits_model[i], total_num_edges[i],
                                       compression_metrics["dictionary_cost"] + parameter_cost)
        dataset_compression.append(bpe_dict_param[1])
        # log for each set in the terminal
        log += '\nEpoch: {:03d}, {} compression_w_dict: {:.4f}, compression: {:.4f} ' \
               ', bpe : {:.2f} bpe_w_dict : {:.2f} bpe_dict_param : {:.2f}'
        log_args += [epoch,
                     dataset_name,
                     compression_metrics["evaluation_metric_w_dict"],
                     compression_metrics["evaluation_metric"],
                     bpe_dict[0], bpe_dict[1], bpe_dict_param[1]]
        # log for each set in wandb after each epoch
        if wandb_realtime:
            if 'empirical_subgraph_types_freqs' in compression_metrics:
                fig = plt.figure()
                plt.bar(list(range(len(compression_metrics["empirical_subgraph_types"]))),
                        compression_metrics["empirical_subgraph_types"])
                wandb.log({"empirical subgraph types freqs {}".format(dataset_name): wandb.Image(fig)}, step=epoch)
                plt.close()
            if 'empirical_atom_probs' in compression_metrics:
                fig = plt.figure()
                plt.bar(list(range(len(compression_metrics["empirical_atom_probs"]))),
                        compression_metrics["empirical_atom_probs"].tolist())
                wandb.log({"empirical atom probabilities {}".format(dataset_name): wandb.Image(fig)}, step=epoch)
                plt.close()
            if 'empirical_b_freqs' in compression_metrics:
                fig = plt.figure()
                plt.bar(list(range(len(compression_metrics["empirical_b_freqs"]))),
                        (compression_metrics["empirical_b_freqs"]/compression_metrics["empirical_b_freqs"].sum()).tolist())
                wandb.log({"empirical b probs".format(dataset_name): wandb.Image(fig)}, step=epoch)
                plt.close()
            log_corpus[dataset_name +  '_bpe_dict_param_fold_'+str(fold_idx)] =  bpe_dict_param[1]
            log_corpus[dataset_name +  '_bpe_dict_fold_'+str(fold_idx)] =  bpe_dict[1]
            log_corpus[dataset_name +  '_bpe_fold_' + str(fold_idx)] = bpe_dict[0]
    # Log for the entire dataset in the terminal
    log += '\nEpoch: {:03d}, num params {}, delta: {:.4f},'
    log_args += [epoch, num_params, compression_metrics["delta_prob"].item()]
    bpe_dict = bits_per_edge(sum(raw_bits_model),
                             sum(total_num_edges),
                             compression_metrics["dictionary_cost"])
    bpe_dict_param = bits_per_edge(sum(raw_bits_model),
                                   sum(total_num_edges),
                                   compression_metrics["dictionary_cost"] + parameter_cost)
    total_compression, total_compression_w_params = bpe_dict_param[0], bpe_dict_param[1]
    log += ', total bpe : {:.2f} total bpe_dict : {:.2f} total bpe_dict_param : {:.2f}'
    log_args += [bpe_dict[0], bpe_dict[1], bpe_dict_param[1]]
    train_compression = dataset_compression[0]
    test_compression = dataset_compression[1] if len(loaders)>1 else None
    val_compression = dataset_compression[2] if len(loaders) > 2 else None
    x_a = compression_metrics["x_a"]
    # Log for the entire dataset in wandb after each epoch
    if wandb_realtime:
        b_probs = compression_metrics["b_probs"]
        delta_prob = compression_metrics["delta_prob"]
        atom_probs = compression_metrics["atom_probs"]
        cut_size_probs = compression_metrics["cut_size_probs"]
        log_corpus['num_params_' + str(fold_idx)] = num_params
        log_corpus['total_bpe_dict_param_fold_' + str(fold_idx)] = bpe_dict_param[1]
        log_corpus['total_bpe_dict_fold_' + str(fold_idx)] = bpe_dict[1]
        log_corpus['total_bpe_fold_' + str(fold_idx)] = bpe_dict[0]
        log_corpus['delta'] = delta_prob.item()
        log_corpus['dictionary_size'] = int(x_a[0:len(agent.env.dictionary)].sum())
        if kwargs['visualise']:
            data_format = 'nx' if  agent.env.isomorphism_module.__class__.__name__ =='ExactIsomoprhism' else 'csr'
            visualise_subgraphs(agent.env.dictionary,
                                init_i=initial_dict_size,
                                final_i=len(agent.env.dictionary),
                                visualise_step=epoch,
                                data_format=data_format,
                                attr_mapping=agent.attr_mapping,
                                node_attr_dims=agent.env.isomorphism_module.node_attr_dims,
                                edge_attr_dims=agent.env.isomorphism_module.edge_attr_dims)

        fig = plt.figure()
        x_a_fractional = dictionary_probs_model.membership_vars(len(agent.env.dictionary)).tolist()[0:len(agent.env.dictionary)]
        plt.bar(list(range(len(agent.env.dictionary))),x_a_fractional)
        wandb.log({"membership fractional variables": wandb.Image(fig)}, step=epoch)
        plt.close()
        fig = plt.figure()
        plt.bar(list(range(len(b_probs))), b_probs.tolist())
        wandb.log({"b probs": wandb.Image(fig)}, step=epoch)
        plt.close()
        if delta_prob is not None:
            fig = plt.figure()
            plt.bar([0,1], [1 - delta_prob.sigmoid().item(), delta_prob.sigmoid().item()])
            wandb.log({"learned subgraph types probs": wandb.Image(fig)}, step=epoch)
            plt.close()
        fig = plt.figure()
        plt.bar(list(range(len(agent.env.dictionary))), atom_probs[:len(agent.env.dictionary)].tolist())
        wandb.log({"learned atom probabilities": wandb.Image(fig)}, step=epoch)
        plt.close()
        if cut_size_probs is not None:
            fig = plt.figure()
            plt.bar(list(range(len(cut_size_probs))), cut_size_probs.tolist())
            wandb.log({"learned cut size probs": wandb.Image(fig)}, step=epoch)
            plt.close()
        fig = plt.figure()
        plt.bar(list(range(len(agent.env.dictionary))),
                agent.env.empirical_atom_freqs[0:len(agent.env.dictionary)].tolist())
        wandb.log({"empirical atom frequencies": wandb.Image(fig)}, step=epoch)
        plt.close()
        wandb.log(log_corpus, step=epoch)
        with open(os.path.join(wandb.run.dir,'selected_atom_inds.txt'), 'a+') as f:
            sorted_probs_inds = torch.argsort(atom_probs[0:len(agent.env.dictionary)], descending=True)
            sorted_mv = x_a[0:len(agent.env.dictionary)][sorted_probs_inds]
            sorted_probs_inds = sorted_probs_inds[sorted_mv==1]
            f.write("Epoch: {}, Atom inds: {}\n".format(epoch, sorted_probs_inds.tolist()))
        wandb.save(os.path.join(wandb.run.dir,'selected_atom_inds.txt'))

    return log, log_args, x_a, \
           train_compression, test_compression, val_compression, \
           total_compression, total_compression_w_params, num_params


def logger(compression_folds, dataset_names, perf_opt, wandb_flag, wandb_realtime):
    datasets_compression_folds, datasets_compression_mean, datasets_compression_std = [], [], []
    for dataset_compression_folds in compression_folds:
        datasets_compression_folds.append(np.array(dataset_compression_folds))
        datasets_compression_mean.append(np.mean(datasets_compression_folds[-1], 0))
        datasets_compression_std.append(np.std(datasets_compression_folds[-1], 0))

    best_index = perf_opt(datasets_compression_mean[dataset_names.index('train')])

    if not wandb_realtime and wandb_flag:
        for epoch in range(len(datasets_compression_mean[0])):
            # log scores for each fold in the current epoch
            log_corpus = {}
            for i, dataset_name in enumerate(dataset_names):
                for fold_idx in range(len(datasets_compression_folds[i])):
                    log_corpus[dataset_name + '_compression_fold_'+str(fold_idx)] = datasets_compression_folds[i][fold_idx, epoch]
                # log epoch score means across folds
                log_corpus[dataset_name+'_compression_mean'] = datasets_compression_mean[i][epoch],
                log_corpus[dataset_name+'_compression_std'] =  datasets_compression_std[i][epoch],
            wandb.log(log_corpus, step=epoch)

    for i, dataset_name in enumerate(dataset_names):
        if wandb_flag:
            wandb.run.summary['best_epoch'] = best_index
            wandb.run.summary['best_'+ dataset_name + '_mean'] = datasets_compression_mean[i][best_index]
            wandb.run.summary['best_'+ dataset_name + '_std'] = datasets_compression_std[i][best_index]
            wandb.run.summary['last_'+ dataset_name + '_mean'] = datasets_compression_mean[i][-1]
            wandb.run.summary['last_'+ dataset_name + '_std'] = datasets_compression_std[i][-1]
        print("Best {} mean: {:.4f} +/- {:.4f}".format(dataset_name,
                                                          datasets_compression_mean[i][best_index],
                                                          datasets_compression_std[i][best_index]))


    return