from utils.misc import isnotebook
if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from utils.visualisation import visualise_subgraphs
epsilon = 1e-5
from utils.loss_evaluation_fns import *
from models.probabilistic_model import *
from utils.test_and_log import test_fixed_part, realtime_logger

def compress_dataset(loader, agent, device, **kwargs):
    subgraphs_all = []
    atom_indices = []
    visualise_dict_all = {}
    for i, data in enumerate(tqdm(loader)):
        data = data.to(device)
        visualise_condition = 'visualise' in kwargs and kwargs['visualise'] and i in kwargs['inds_to_visualise']
        kwargs_visualise_i = {'visualise': visualise_condition,
                                'inds_to_visualise': [0],
                                'visualise_step': 0} if visualise_condition else {}
        subgraphs, visualise_dict = agent.compress(data, **kwargs_visualise_i)
        subgraphs_all.append(subgraphs)

        atom_indices_batch = subgraphs['atom_indices'].flatten()
        atom_indices_batch = atom_indices_batch[atom_indices_batch != -1]
        atom_indices.append(atom_indices_batch)
        if visualise_condition:
            visualise_dict_all[i] = visualise_dict[0]
    estimated_atom_freqs, estimated_atom_probs = agent.env.estimate_atom_freqs_probs(torch.cat(atom_indices))

    return subgraphs_all, estimated_atom_probs, visualise_dict_all




def train(agent,
          dictionary_probs_model,
          loader_train,
          loader_test,
          optim_phi,
          start_epoch,
          n_epochs,
          eval_freq=1,
          loader_val=None,
          evaluation_fn=None,
          scheduler_phi=None,
          min_lr=0.0,
          checkpoint_file=None,
          wandb_realtime=False,
          fold_idx=-1,
          mode='train',
          **kwargs):

    if wandb_realtime and kwargs['visualise']:
        data_format = 'nx' if  agent.env.isomorphism_module.__class__.__name__ =='ExactIsomoprhism' else 'csr'
        visualise_subgraphs(agent.env.dictionary,
                            init_i=0,
                            final_i=len(agent.env.dictionary),
                            visualise_step=0,
                            data_format=data_format,
                            attr_mapping = agent.attr_mapping,
                            node_attr_dims = agent.env.isomorphism_module.node_attr_dims,
                            edge_attr_dims = agent.env.isomorphism_module.edge_attr_dims)

    train_compression_p_epoch = []
    test_compression_p_epoch = []
    val_compression_p_epoch = []
    total_compression_p_epoch = []
    total_compression_w_params_p_epoch = []
    num_params_p_epoch = []

    device = agent.env.device
    initial_dict_size = 0
    kwargs_visualise = {'visualise': kwargs['visualise'] if 'visualise' in kwargs else False,
                        'inds_to_visualise':kwargs['inds_to_visualise'] if 'inds_to_visualise' in kwargs else 0,
                        'visualise_step': kwargs['visualise_step'] if 'visualise_step' in kwargs else 0}

    loaders = [loader_train]
    dataset_names = ['train']
    agent.train = True
    subgraphs_train, estimated_atom_probs_train, _ = compress_dataset(loader_train, agent, device, **kwargs_visualise)
    subgraphs_all = [subgraphs_train]
    if loader_test is not None:
        loaders.append(loader_test)
        dataset_names += ['test']
        agent.train = False
        subgraphs_test, estimated_atom_probs_test, visualise_data_dict = compress_dataset(loader_test, agent, device, **kwargs_visualise)
        subgraphs_all.append(subgraphs_test)

    if loader_val is not None:
        loaders.append(loader_val)
        dataset_names += ['val']
        agent.train = False
        subgraphs_val, estimated_atom_probs_val,_ = compress_dataset(loader_val, agent, device)
        subgraphs_all.append(subgraphs_val)

    if wandb_realtime:
        fig = plt.figure()
        plt.bar(list(range(len(estimated_atom_probs_train))),
                estimated_atom_probs_train.tolist())
        wandb.log({"empirical atom probabilities train": wandb.Image(fig)}, step=0)
        plt.close()
        if loader_test is not None:
            fig = plt.figure()
            plt.bar(list(range(len(estimated_atom_probs_test))),
                    estimated_atom_probs_test.tolist())
            wandb.log({"empirical atom probabilities test": wandb.Image(fig)}, step=0)
            plt.close()
        if loader_val is not None:
            fig = plt.figure()
            plt.bar(list(range(len(estimated_atom_probs_val))),
                    estimated_atom_probs_val.tolist())
            wandb.log({"empirical atom probabilities val": wandb.Image(fig)}, step=0)
            plt.close()

    best_ref_metric = None
    if mode == 'test':
        start_epoch, n_epochs, n_iters = 0, 1, 0
    for epoch in tqdm(range(start_epoch, n_epochs)):
        if mode=='train':
            agent.train = True
            optim_phi.zero_grad()
            x_a, b_probs, delta_prob, atom_probs, cut_size_probs = dictionary_probs_model.prune_universe(agent.train)
            cost_graphs,_ = ml_estimation_fixed_part(subgraphs_train, agent,
                                                     x_a, b_probs, delta_prob, atom_probs, cut_size_probs)
            dataset_cost = cost_graphs.sum()
            atom_costs = torch.zeros_like(x_a)
            atom_costs[0:len(agent.env.dictionary)] = agent.env.compute_atom_costs()
            optimise_dict_prob_model(optim_phi,
                                     dataset_cost,
                                     atom_costs,
                                     x_a,
                                     kwargs['amortisation_param'])
            for scheduler in [scheduler_phi]:
                if scheduler and 'ReduceLROnPlateau' not in str(scheduler.__class__):
                    scheduler.step()
        if epoch % eval_freq == 0 or epoch==n_epochs-1:
            with torch.no_grad():
                kwargs['visualise_step'] = epoch
                log, log_args, x_a, \
                train_comp, test_comp, val_comp, \
                total_comp, total_comp_w_params, num_params = realtime_logger(loaders,
                                                                              dataset_names,
                                                                              agent,
                                                                              dictionary_probs_model,
                                                                              evaluation_fn,
                                                                              epoch,
                                                                              fold_idx,
                                                                              wandb_realtime,
                                                                              initial_dict_size,
                                                                              test_fn=test_fixed_part,
                                                                              subgraphs_all=subgraphs_all,
                                                                              visualise_data_dict=visualise_data_dict,
                                                                              n_iters_test=None,
                                                                              **kwargs)
                if mode == 'train':
                    log +=   ',dictionary probs lr: {:.8f}'
                    log_args += [optim_phi.param_groups[0]['lr']]
                log +=   ', current dictionary size: {:04d}, total dictionary size: {:04d}'
                log_args += [int(x_a[0:len(agent.env.dictionary)].sum()), len(agent.env.dictionary)]
                print(log.format(*log_args))
                num_params_p_epoch.append(num_params)
                train_compression_p_epoch.append(train_comp)
                total_compression_p_epoch.append(total_comp)
                total_compression_w_params_p_epoch.append(total_comp_w_params)
                if test_comp is not None:
                    test_compression_p_epoch.append(test_comp)
                if val_comp is not None:
                    val_compression_p_epoch.append(val_comp)
            if mode == 'train':
                ref_metric = val_comp if loader_val is not None else test_comp
                for scheduler in [scheduler_phi]:
                    if scheduler and 'ReduceLROnPlateau' in str(scheduler.__class__):
                        scheduler.step(ref_metric)
                if best_ref_metric is None or ref_metric < best_ref_metric:
                    scheduler_phi_state_dict = scheduler_phi.state_dict() if scheduler_phi is not None else None
                    torch.save({'epoch': epoch,
                        'dictionary_probs_state_dict': dictionary_probs_model.state_dict(),
                        'optimizer_phi_state_dict' : optim_phi.state_dict(),
                        'scheduler_phi_state_dict': scheduler_phi_state_dict,
                        'dictionary': (agent.env.dictionary, agent.env.dictionary_num_vertices, agent.env.dictionary_num_edges),
                        }, checkpoint_file)
        if mode == 'train':
            current_lr = optim_phi.param_groups[0]['lr']
            if current_lr < min_lr:
                break
        initial_dict_size = len(agent.env.dictionary)
    test_compression_p_epoch = None if len(test_compression_p_epoch) == 0 else test_compression_p_epoch
    val_compression_p_epoch = None if len(val_compression_p_epoch) == 0 else val_compression_p_epoch
    return train_compression_p_epoch, test_compression_p_epoch, \
           val_compression_p_epoch, total_compression_p_epoch, \
           total_compression_w_params_p_epoch, num_params_p_epoch