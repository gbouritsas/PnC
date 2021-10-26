import torch

def expected_return(compressed_size, uncompressed_size):
    returns = uncompressed_size - compressed_size
    return returns.mean()

def expected_space_saving(compressed_size, uncompressed_size):
    return (1 - compressed_size/uncompressed_size).mean()

def dataset_space_saving(compressed_size, uncompressed_size, model_cost):
    return (1 - (compressed_size.sum() + model_cost)/uncompressed_size.sum()).item(), \
           (1 - compressed_size.sum()/uncompressed_size.sum()).item()

def expected_compression_ratio(compressed_size, uncompressed_size):
    return (uncompressed_size/compressed_size).mean()

def dataset_compression_ratio(compressed_size, uncompressed_size):
    return uncompressed_size.sum()/compressed_size.sum()

def bits_per_edge(compressed_size, num_edges, model_cost):
    return compressed_size/num_edges, (compressed_size + model_cost)/num_edges

def compute_entropy(probs):
    probs = probs[probs != 0]
    return (- probs * torch.log2(probs)).sum()

def optimise_dict_prob_model(optim_phi,
                             dataset_cost,
                             atom_costs,
                             x_a,
                             amortisation_param=1):
    phi_loss = (atom_costs *  x_a).sum() + amortisation_param * dataset_cost
    phi_loss.backward()
    optim_phi.step()

    return


def policy_gradient_update(optim_theta, log_probs, total_costs, baseline, samples=None, dataset_size=1):
    if samples!=0:
        # print((total_costs.detach() - baseline).sum() / s_a_pairs)
        theta_loss = dataset_size * (log_probs * (total_costs.detach() - baseline)).sum() / samples
        theta_loss.backward()
        optim_theta.step()
    return

def ml_estimation_fixed_part(subgraphs_all, agent,
                             x_a, b_probs, delta_prob, atom_probs, cut_size_probs=None):
    cost_graphs = []
    cost_baseline = []
    for subgraphs in subgraphs_all:
        total_costs, baseline, _, _ = agent.env.compute_dl(subgraphs,
                                                           x_a,
                                                           b_probs,
                                                           delta_prob,
                                                           atom_probs,
                                                           cut_size_probs,
                                                           log_probs=None)

        cost_graphs.append(total_costs)
        cost_baseline.append(baseline)

    return torch.cat(cost_graphs), torch.cat(cost_baseline)



def ml_estimation_neural_part(data, agent, optim_theta,
                              x_a, b_probs, delta_prob, atom_probs, cut_size_probs,
                              device,
                              update_theta=False,
                              dataset_size=1):

    data = data.to(device)
    if update_theta:
        optim_theta.zero_grad()
    subgraphs, log_probs_actions,_ = agent.compress(data)
    total_costs, baseline, log_probs, _ = agent.env.compute_dl(subgraphs,
                                                               x_a,
                                                               b_probs,
                                                               delta_prob,
                                                               atom_probs,
                                                               cut_size_probs,
                                                               log_probs_actions)
    samples = total_costs.shape[0]
    if update_theta:
        policy_gradient_update(optim_theta,
                               log_probs,
                               total_costs,
                               baseline,
                               samples=samples,
                               dataset_size=dataset_size)
    # update empirical freqs/probs for faster subgraph retrieval
    atom_indices_batch = subgraphs['atom_indices'].flatten()
    atom_indices_batch = atom_indices_batch[atom_indices_batch!=-1]
    estimated_atom_freqs, estimated_atom_probs = agent.env.estimate_atom_freqs_probs(atom_indices_batch)
    _,_ = agent.env.update_empirical_atom_freqs_probs(estimated_atom_freqs, estimated_atom_probs, save_update=True)
    dataset_cost = dataset_size * total_costs.mean()
    return dataset_cost
