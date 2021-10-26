import sys
sys.path.append('../')
import argparse
import utils.parsing as parse
import os
import torch
import numpy as np
import random
from utils.prepare_data import prepare_dataset, prepape_input_features, prepare_dataloaders
from utils.prepare_dictionary import prepare_dictionary
from utils.isomorphism_modules import prepare_isomorphism_module
from utils.prepare_arguments import prepare_environment_args
from encoding_decoding.environment import CompressionEnvironment
from models.probabilistic_model import ProbabilisticModel
from agent_fixed_part import CompressionAgentFixedPart
from utils.optim import setup_optimization, resume_training_phi
import utils.loss_evaluation_fns as loss_evaluation_fns
from utils.test_and_log import prepare_logs, logger
from train_fixed_part import train


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
    if args['GPU']:
        device = torch.device("cuda:"+str(args['device_idx']) if torch.cuda.is_available() else "cpu")
        print('[info] Training will be performed on {}'.format(torch.cuda.get_device_name(args['device_idx'])))
    else:
        device = torch.device("cpu")
        print('[info] Training will be performed on cpu')
    if args['wandb']:
        print('Wandb logging activated... ')
        import wandb
        wandb.init(sync_tensorboard=False, project=args['wandb_project'],
                   reinit = False, config = args, entity=args['wandb_entity'])
        print('[info] Monitoring with wandb')
    path = os.path.join(args['root_folder'], args['dataset'], args['dataset_name'])
    perf_opt = np.argmin
    train_compression_folds = []
    test_compression_folds = []
    val_compression_folds = []
    total_compression_folds = []
    total_compression_w_params_folds = []
    num_params_folds = []
    fold_idxs = args['fold_idx']
    assert args['mode'] in ['train','test'], "Unknown mode. Supported options are  'train','test'"
    for fold_idx in fold_idxs:
        print('############# FOLD NUMBER {:01d} #############'.format(fold_idx))
        ## ----------------------------------- preparation of model components
        # prepare dictionary with initial dictionary atoms (optional)
        if 'motifs' in args['atom_types']:
            graphs_ptg = prepare_dataset(path,
                                         args['dataset'],
                                         args['dataset_name'],
                                         directed=args['directed'])
            loader_train, _, _ = prepare_dataloaders(args, graphs_ptg, path, fold_idx, False, 0)
            dataset_train = loader_train.dataset
            if args['split'] == 'random':
                split_folder = 'split_idx_random_' + str(args['split_seed']) + '/' + str(fold_idx)
            else:
                split_folder = 'split_idx' + '/' + str(fold_idx)
            ## ----------------------------------- preparation of model components
            # prepare dictionary
            H_set_gt = prepare_dictionary(args, path=path, graphs_ptg=dataset_train, split_folder=split_folder)
            args['max_dict_size'] = len(H_set_gt)
        else:
            H_set_gt = prepare_dictionary(args)
        # generate/load dataset + detect/load subgraphs
        graphs_ptg = prepare_dataset(path,
                                     args['dataset'],
                                     args['dataset_name'],
                                     directed=args['directed'],
                                     H_set=H_set_gt,
                                     multiprocessing=args['multiprocessing'],
                                     num_processes=args['num_processes'],
                                     candidate_subgraphs=args['candidate_subgraphs'])
        # one-hot encoding etc of input features
        graphs_ptg, in_features_dims_dict, attr_mapping = prepape_input_features(args, graphs_ptg, path)
        # prepare and instantiate isomoprhism module
        isomorphism_module = prepare_isomorphism_module(args['isomorphism_type'],
                                                        node_attr_dims=None if args['node_attr_encoding'] is None
                                                        else attr_mapping.node_attr_dims,
                                                        edge_attr_dims=None if args['edge_attr_encoding'] is None
                                                        else attr_mapping.edge_attr_dims)
        # prepare compression environment
        environment_args = prepare_environment_args(args,
                                                    graphs_ptg,
                                                    H_set_gt,
                                                    device,
                                                    isomorphism_module,
                                                    in_features_dims_dict['node_attr_unique_values'],
                                                    in_features_dims_dict['edge_attr_unique_values'])
        environment = CompressionEnvironment(**environment_args)
        print('Num graphs: {}'.format(len(graphs_ptg)))
        print('Max degree: {}'.format(in_features_dims_dict['degree_unique_values'][0] - 1))
        n_mean = np.mean([graph.x.shape[0] for graph in graphs_ptg])
        print('Avg/Max num nodes: {:.2f}, {}'.format(n_mean , environment_args['n_max']))
        ## ----------------------------------- prepare model (agent)
        kwargs_agent = {'n_h_max': args['n_h_max'],
                        'n_h_min': args['n_h_min'],
                        'n_h_max_dict': args['n_h_max_dict'],
                        'n_h_min_dict': args['n_h_min_dict'],
                        'attr_mapping': attr_mapping}
        agent = CompressionAgentFixedPart(environment, partitioning_algorithm=args['partitioning_algorithm'], **kwargs_agent)
        ## ----------------------------------- prepare evaluators and loggers
        evaluation_fn = getattr(loss_evaluation_fns, args['evaluation_fn'])
        checkpoint_folder = prepare_logs(args, path, fold_idx)
        # prepare dataloaders
        loader_train, loader_test, loader_val = prepare_dataloaders(args,
                                                                    graphs_ptg,
                                                                    path,
                                                                    fold_idx,
                                                                    args['candidate_subgraphs'],
                                                                    len(H_set_gt))
        # initialise dictionary and probabilistic model learnable parameters
        dictionary_probs_model = ProbabilisticModel(args['max_dict_size'],
                                                  args['b_max'],
                                                  b_distribution=args['b_distribution'],
                                                  delta_distribution=args['delta_distribution'],
                                                  atom_distribution=args['atom_distribution'],
                                                  cut_size_distribution=args['cut_size_distribution'],
                                                  b_min=args['b_min'],
                                                  c_max=None).to(device)
        print("Instantiated model:\n{}".format(dictionary_probs_model))
        # count model params
        params = sum(p.numel() for p in dictionary_probs_model.parameters() if p.requires_grad)
        print("[info] Total number of parameters is: {}".format(params))
        kwargs_train_test = {'visualise': args['visualise'] and args['wandb'],
                             'inds_to_visualise': [] if args['inds_to_visualise'] is None else args['inds_to_visualise'],
                             'bits_per_parameter': args['bits_per_parameter'],
                             'amortisation_param': args['amortisation_param']}
        if args['mode'] == 'train':
            print("Training starting now...")
            # optimizer and lr scheduler
            trainable_parameters_phi = [p for p in dictionary_probs_model.parameters() if p.requires_grad]
            kargs_optim_phi = {'lr': args['lr_dict'],
                               'regularization': args['regularization'],
                               'scheduler': args['scheduler'],
                               'scheduler_mode': args['scheduler_mode'],
                               'decay_rate': args['decay_rate'],
                               'decay_steps': args['decay_steps'],
                               'patience': args['patience']}
            optim_phi, scheduler_phi = setup_optimization(trainable_parameters_phi, **kargs_optim_phi)
        else:
            optim_phi, scheduler_phi = None, None
        if args['resume']:
            resume_folder = args['resume_folder'] if args['resume_folder'] is not None else checkpoint_folder
            resume_filename = os.path.join(resume_folder, args['checkpoint_file'] + '.pth.tar')
            start_epoch = resume_training_phi(resume_filename,
                                              dictionary_probs_model,
                                              optim_phi,
                                              scheduler_phi,
                                              device,
                                              env=agent.env)
        else:
            start_epoch = 0
        # logging
        if args['wandb']:
            wandb.watch(dictionary_probs_model)
        checkpoint_filename = os.path.join(checkpoint_folder, args['checkpoint_file'] + '.pth.tar')
        # train (!)
        metrics = train(agent,
                        dictionary_probs_model,
                        loader_train,
                        loader_test,
                        optim_phi,
                        start_epoch=start_epoch,
                        n_epochs=args['num_epochs'],
                        eval_freq=args['eval_frequency'],
                        loader_val=loader_val,
                        evaluation_fn=evaluation_fn,
                        scheduler_phi=scheduler_phi,
                        min_lr=args['min_lr'],
                        checkpoint_file=checkpoint_filename,
                        wandb_realtime=args['wandb_realtime'] and args['wandb'],
                        fold_idx=fold_idx,
                        mode=args['mode'],
                        **kwargs_train_test)
        print("Training/Testing complete!")
        # log results of training
        train_compression_p_epoch, test_compression_p_epoch, val_compression_p_epoch, \
        total_compression_p_epoch, total_compression_w_params_p_epoch, num_params_p_epoch = metrics
        train_compression_folds.append(train_compression_p_epoch)
        test_compression_folds.append(test_compression_p_epoch)
        val_compression_folds.append(val_compression_p_epoch)
        total_compression_folds.append(total_compression_p_epoch)
        total_compression_w_params_folds.append(total_compression_w_params_p_epoch)
        num_params_folds.append(num_params_p_epoch)
        best_idx = perf_opt(train_compression_p_epoch)
        print("\tbest epoch {}\n\tbest train accuracy {:.4f}, "
              "\n\tbest test accuracy {:.4f},"
              "\n\tbest total accuracy {:.4f},"
              " \n\tbest total w params accuracy {:.4f}, "
              " \n\tnum params {},".
              format(best_idx,
                     train_compression_p_epoch[best_idx],
                     test_compression_p_epoch[best_idx],
                     total_compression_p_epoch[best_idx],
                     total_compression_w_params_p_epoch[best_idx],
                     num_params_p_epoch[best_idx]))

    if args['mode'] == 'train':
        if loader_val is not None:
            compression_folds = [train_compression_folds, test_compression_folds,
                                 val_compression_folds, total_compression_folds,
                                 total_compression_w_params_folds, num_params_folds]
            names = ['train', 'test', 'val', 'total', 'total_w_params', 'num_params']
        elif loader_test is not None:
            compression_folds = [train_compression_folds, test_compression_folds,
                                 total_compression_folds, total_compression_w_params_folds,
                                 num_params_folds]
            names = ['train', 'test', 'total', 'total_w_params', 'num_params']
        else:
            compression_folds = [train_compression_folds, total_compression_folds,
                                 total_compression_w_params_folds, num_params_folds]
            names = ['train', 'total', 'total_w_params', 'num_params']
        logger(compression_folds,
               names,
               perf_opt,
               args['wandb'],
               args['wandb_realtime'])

    if args['mode'] == 'test':
        print("Train accuracy: {:.4f} +/- {:.4f}".format(np.mean(train_compression_folds), np.std(train_compression_folds)))
        print("Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_compression_folds), np.std(test_compression_folds)))
        if loader_val is not None:
            print("Validation accuracy: {:.4f} +/- {:.4f}".format(np.mean(val_compression_folds), np.std(val_compression_folds)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #----------------- seeds
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--np_seed', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=1)
    #----------------- infrastructure + dataloader + logging + visualisation
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--wandb', type=parse.str2bool, default=False)
    parser.add_argument('--wandb_realtime', type=parse.str2bool, default=False)
    parser.add_argument('--wandb_project', type=str, default="graph_compression")
    parser.add_argument('--wandb_entity', type=str, default="epfl")
    parser.add_argument('--visualise', type=parse.str2bool, default=False)
    parser.add_argument('--inds_to_visualise', type=parse.str2list2int, default=None)
    parser.add_argument('--GPU', type=parse.str2bool, default=True)
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--results_folder', type=str, default='PnC_fixed_part_experiments')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint')
    parser.add_argument('--resume', type=parse.str2bool, default=False)
    parser.add_argument('--resume_folder', type=str, default=None)  # useful to load models trained on other distributions
    #----------------- dataset and split
    parser.add_argument('--root_folder', type=str, default='../datasets/')
    parser.add_argument('--dataset', type=str, default='bioinformatics')
    parser.add_argument('--dataset_name', type=str, default='MUTAG')
    parser.add_argument('--directed', type=parse.str2bool, default=False)
    parser.add_argument('--fold_idx', type=parse.str2list2int, default=[0])
    parser.add_argument('--split', type=str, default='given')
    parser.add_argument('--split_seed', type=int, default=0) # only for random splitting
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    #----------------- compression environment
    parser.add_argument('--universe_type', type=str, default='adaptive')
    parser.add_argument('--max_dict_size', type=int, default=10000)
    # constants
    parser.add_argument('--precision', type=int, default=None)
    parser.add_argument('--n_max', type=int, default=None)
    parser.add_argument('--e_max', type=int, default=None)
    parser.add_argument('--d_max', type=int, default=None)
    parser.add_argument('--c_max', type=int, default=None)
    parser.add_argument('--b_max', type=int, default=None)
    parser.add_argument('--b_min', type=int, default=None)
    parser.add_argument('--n_h_max_dict', type=int, default=-1) # -1 means that n_h_max_dict = n_max
    parser.add_argument('--n_h_min_dict', type=int, default=1)
    parser.add_argument('--n_h_max', type=int, default=-1)
    parser.add_argument('--n_h_min', type=int, default=1)
    # ----------------- encoding schemes
    # dictionary encoding
    parser.add_argument('--dictionary_encoding', type=str, default='graphs')
    parser.add_argument('--num_nodes_atom_encoding', type=str, default='uniform')
    parser.add_argument('--num_edges_atom_encoding', type=str, default='uniform')
    parser.add_argument('--adj_matrix_atom_encoding', type=str, default='erdos_renyi')
    # subgraph encoding (dictionary and non-dictionary)
    parser.add_argument('--dict_subgraphs_encoding', type=str, default='multinomial')
    parser.add_argument('--num_nodes_encoding', type=str, default='uniform')
    parser.add_argument('--num_edges_encoding', type=str, default='uniform')
    parser.add_argument('--adj_matrix_encoding', type=str, default='erdos_renyi')
    # cut encoding
    parser.add_argument('--cut_encoding', type=str, default='joint')
    parser.add_argument('--cut_size_encoding', type=str, default='uniform')
    parser.add_argument('--cut_edges_encoding', type=str, default='erdos_renyi')
    # attribute encoding
    parser.add_argument('--node_attr_encoding', type=str, default=None)
    parser.add_argument('--edge_attr_encoding', type=str, default=None)
    # baseline encoding
    parser.add_argument('--num_nodes_baseline_encoding', type=str, default='uniform')
    parser.add_argument('--num_edges_baseline_encoding', type=str, default='uniform')
    parser.add_argument('--adj_matrix_baseline_encoding', type=str, default='erdos_renyi')
    # ----------------- isomorphism
    parser.add_argument('--isomorphism_type', type=str, default='exact')
    # exponential moving average coefficient: used to compute the empirical frequencies of the atoms in order
    # to speed-up the matching between subgraphs and dictionary atoms
    parser.add_argument('--ema_coeff', type=float, default=0.5)
    # ----------------- probabilistic model learnable parameters
    parser.add_argument('--b_distribution', type=str, default='learnable')
    parser.add_argument('--delta_distribution', type=str, default='learnable')
    parser.add_argument('--atom_distribution', type=str, default='learnable')
    parser.add_argument('--cut_size_distribution', type=str, default='uniform')
    # ----------------- optimisation and learning parameters
    parser.add_argument('--shuffle', type=parse.str2bool, default=False)
    parser.add_argument('--eval_frequency', type=int, default=1)
    parser.add_argument('--regularization', type=float, default=0)
    parser.add_argument('--scheduler', type=str, default='None')
    parser.add_argument('--scheduler_mode', type=str, default='min')
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--decay_steps', type=int, default=50)
    parser.add_argument('--decay_rate', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--amortisation_param', type=float, default=1)
    parser.add_argument('--evaluation_fn', type=str, default='dataset_space_saving')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr_dict', type=float, default=1)

    parser.add_argument('--partitioning_algorithm', type=str, default='Louvain')
    parser.add_argument('--bits_per_parameter', type=int, default=16)

    # ----------------- Special cases (fixed dictionary and/or candidate subgraphs)
    # initial dictionary (optional):
    parser.add_argument('--atom_types', type=parse.str2list2str, default=[])
    parser.add_argument('--k', type=parse.str2list2int, default=[])
    parser.add_argument('--custom_edge_lists', type=parse.str2ListOfListsOfLists2int, default=None)
    # subgraph isomorphism (optional):
    parser.add_argument('--candidate_subgraphs', type=parse.str2bool, default=False)
    parser.add_argument('--multiprocessing', type=parse.str2bool, default=False)
    parser.add_argument('--num_processes', type=int, default=64)

    args = parser.parse_args()
    print(args)
    main(vars(args))



