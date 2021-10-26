from utils.misc import isnotebook
if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
from utils.visualisation import visualise_subgraphs
epsilon = 1e-5
from utils.loss_evaluation_fns import *
from models.probabilistic_model import *
from utils.test_and_log import test_neural_part, realtime_logger
import copy

def train(agent,
          dictionary_probs_model,
          loader_train,
          loader_test,
          optim_phi,
          optim_theta,
          start_epoch,
          n_epochs,
          n_iters=None,
          n_iters_test=None,
          eval_freq=1,
          loader_val=None,
          evaluation_fn=None,
          scheduler_phi=None,
          scheduler_theta=None,
          min_lr=0.0,
          checkpoint_file=None,
          wandb_realtime=False,
          fold_idx=-1,
          mode='train',
          **kwargs):

    loaders = [loader_train]
    dataset_names = ['train']
    if loader_test is not None:
        loaders += [loader_test]
        dataset_names += ['test']
    if loader_val is not None:
        loaders += [loader_val]
        dataset_names += ['val']

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

    # NOTE: you might want to change early termination between train and test
    train_compression_p_epoch = []
    test_compression_p_epoch = []
    val_compression_p_epoch = []
    total_compression_p_epoch = []
    total_compression_w_params_p_epoch = []
    num_params_p_epoch = []
    best_ref_metric = None
    device = agent.env.device
    dataset_size = len(loader_train.dataset)
    if mode == 'test':
        start_epoch, n_epochs, n_iters = 0, 1, 0
    for epoch in range(start_epoch, n_epochs):
        agent.train = True
        initial_dict_size = len(agent.env.dictionary)
        data_iterator = iter(loader_train)
        n_iters = len(loader_train) if n_iters is None else n_iters
        for _ in tqdm(range(n_iters)):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader_train)
                data = next(data_iterator)
            optim_phi.zero_grad()
            x_a, b_probs, delta_prob, atom_probs, cut_size_probs = dictionary_probs_model.prune_universe(agent.train)
            # estimate the likelihood
            dataset_cost = ml_estimation_neural_part(data, agent, optim_theta,
                                                     x_a, b_probs, delta_prob, atom_probs, cut_size_probs,
                                                     device,
                                                     update_theta=True,
                                                     dataset_size=dataset_size)

            atom_costs = torch.zeros_like(x_a)
            atom_costs[0:len(agent.env.dictionary)] = agent.env.compute_atom_costs()
            optimise_dict_prob_model(optim_phi,
                                     dataset_cost,
                                     atom_costs,
                                     x_a,
                                     kwargs['amortisation_param'])

        if mode == 'train':
            for scheduler in [scheduler_phi, scheduler_theta]:
                 if scheduler and 'ReduceLROnPlateau' not in str(scheduler.__class__):
                    scheduler.step()
        if epoch % eval_freq == 0 or epoch==n_epochs-1:
            with torch.no_grad():
                kwargs['visualise_step'] = epoch
                if kwargs['bits_per_parameter'] == 16:
                    model_to_send = copy.deepcopy(dictionary_probs_model)
                    model_to_send.b_logits = torch.nn.Parameter(model_to_send.b_logits.half().float())
                    model_to_send.subgraph_in_dict_logit = torch.nn.Parameter(model_to_send.subgraph_in_dict_logit.half().float())
                    model_to_send.atom_logits = torch.nn.Parameter(model_to_send.atom_logits.half().float())
                else:
                    model_to_send = dictionary_probs_model
                log, log_args, x_a, train_comp, \
                test_comp, val_comp, total_comp, \
                total_comp_w_params, num_params = realtime_logger(loaders,
                                                                  dataset_names,
                                                                  agent,
                                                                  model_to_send,
                                                                  # dictionary_probs_model,
                                                                  evaluation_fn,
                                                                  epoch,
                                                                  fold_idx,
                                                                  wandb_realtime,
                                                                  initial_dict_size,
                                                                  test_fn=test_neural_part,
                                                                  subgraphs_all=None,
                                                                  visualise_data_dict=None,
                                                                  n_iters_test=n_iters_test,
                                                                  **kwargs)
                if mode == 'train':
                    log +=   '\nGNN lr: {:.8f}'
                    log_args += [optim_theta.param_groups[0]['lr']]
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
                ref_metric = val_comp if loader_val is not None else train_comp
                for scheduler in [scheduler_phi, scheduler_theta]:
                    if scheduler and 'ReduceLROnPlateau' in str(scheduler.__class__):
                        scheduler.step(ref_metric)
                if best_ref_metric is None or ref_metric < best_ref_metric :
                    best_ref_metric = ref_metric
                    scheduler_phi_state_dict = scheduler_phi.state_dict() if scheduler_phi is not None else None
                    scheduler_theta_state_dict = scheduler_theta.state_dict() if scheduler_theta is not None else None
                    torch.save({'epoch': epoch,
                        'dictionary_probs_state_dict': dictionary_probs_model.state_dict(),
                        'policy_state_dict': agent.policy.state_dict(),
                        'optimizer_phi_state_dict' : optim_phi.state_dict(),
                        'optimizer_theta_state_dict': optim_theta.state_dict(),
                        'scheduler_phi_state_dict': scheduler_phi_state_dict,
                        'scheduler_theta_state_dict': scheduler_theta_state_dict,
                        'dictionary': (agent.env.dictionary, agent.env.dictionary_num_vertices, agent.env.dictionary_num_edges),
                        }, checkpoint_file)
        if mode == 'train':
            current_lr = min(optim_phi.param_groups[0]['lr'], optim_theta.param_groups[0]['lr'])
            if current_lr < min_lr:
                break
    test_compression_p_epoch = None if len(test_compression_p_epoch) == 0 else test_compression_p_epoch
    val_compression_p_epoch = None if len(val_compression_p_epoch) == 0 else val_compression_p_epoch
    return train_compression_p_epoch, test_compression_p_epoch, \
           val_compression_p_epoch, total_compression_p_epoch, \
           total_compression_w_params_p_epoch, num_params_p_epoch
