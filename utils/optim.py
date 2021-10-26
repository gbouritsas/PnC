import torch

def setup_optimization(parameters, optimiser_name='Adam',**args):
    # -------------- instantiate optimizer and scheduler

    optimizer = getattr(torch.optim, optimiser_name)(parameters, lr=args['lr'], weight_decay=args['regularization'])
    if args['scheduler'] == 'ReduceLROnPlateau':
        print("Instantiating ReduceLROnPlateau scheduler.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args['scheduler_mode'],
                                                               factor=args['decay_rate'],
                                                               patience=args['patience'],
                                                               verbose=True)
    elif args['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args['decay_steps'], gamma=args['decay_rate'])
    elif args['scheduler'] == 'None':
        scheduler = None
    else:
        raise NotImplementedError('Scheduler {} is not currently supported.'.format(args['scheduler']))
    return optimizer, scheduler


def resume_training_phi(checkpoint_filename,
                        dictionary_probs_model,
                        optim_phi,
                        scheduler_phi,
                        device,
                        env=None):
    print('Loading checkpoint from file {}'.format(checkpoint_filename))
    checkpoint_dict = torch.load(checkpoint_filename, map_location=device)
    start_epoch = checkpoint_dict['epoch'] + 1
    dictionary_probs_model.load_state_dict(checkpoint_dict['dictionary_probs_state_dict'])
    if optim_phi is not None:
        optim_phi.load_state_dict(checkpoint_dict['optim_phi_state_dict'])
    if scheduler_phi is not None:
        scheduler_phi.load_state_dict(checkpoint_dict['scheduler_phi_state_dict'])
    env.dictionary, env.dictionary_num_vertices, env.dictionary_num_edges = checkpoint_dict['dictionary']
    print('Resuming from epoch {}'.format(start_epoch))
    return start_epoch


def resume_training_phi_theta(checkpoint_filename,
                              policy_network,
                              dictionary_probs_model,
                              optim_phi,
                              optim_theta,
                              scheduler_phi,
                              scheduler_theta,
                              device,
                              env=None):
    print('Loading checkpoint from file {}'.format(checkpoint_filename))
    checkpoint_dict = torch.load(checkpoint_filename, map_location=device)
    start_epoch = checkpoint_dict['epoch'] + 1
    policy_network.load_state_dict(checkpoint_dict['policy_state_dict'])
    dictionary_probs_model.load_state_dict(checkpoint_dict['dictionary_probs_state_dict'])
    if optim_phi is not None:
        optim_phi.load_state_dict(checkpoint_dict['optimizer_phi_state_dict'])
    if optim_theta is not None:
        optim_theta.load_state_dict(checkpoint_dict['optimizer_theta_state_dict'])
    if scheduler_phi is not None:
        scheduler_phi.load_state_dict(checkpoint_dict['scheduler_phi_state_dict'])
    if scheduler_theta is not None:
        scheduler_theta.load_state_dict(checkpoint_dict['scheduler_theta_state_dict'])
    env.dictionary, env.dictionary_num_vertices, env.dictionary_num_edges = checkpoint_dict['dictionary']
    print('Resuming from epoch {}'.format(start_epoch))
    return start_epoch