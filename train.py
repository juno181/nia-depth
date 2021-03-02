""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import time
import sys

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import utils
import losses
import random


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['M_activation'] = utils.activation_dict[config['M_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    if config['seed'] == 0:
        config['seed'] = random.randint(0, 2**32 - 1)
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    sys.path.insert(0, './models')
    model = __import__(config['model'])
    print('Import model from: ', config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    M = model.model(**config).to(device)

    # FP16?
    if config['M_fp16']:
        print('Casting M to float16...')
        M = M.half()
    M_par = model.parallel(M)
    print(M)
    print('Number of params in M: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [M]]))

    # Set loss
    loss = losses.MaskedMSELoss()

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_loss': 9999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(M, state_dict,
                           config['weights_root'], config['resume_experiment_name'])

    # If parallel, parallelize the GD module
    if config['parallel']:
        M_par = nn.DataParallel(M_par)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                              experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Test Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                   reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                               reinitialize=(not config['resume']),
                               logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)


    if config['save_code']:
        # copy codes and config file
        output_dir = '%s/%s' % (config['logs_root'], experiment_name)
        files = utils.list_dir_recursively_with_ignore('.', ignores=['diagrams', 'configs', 'samples', 'logs', 'imgs', 'weights', ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*"])
        files = [(f[0], os.path.join(output_dir, "src", f[1])) for f in files]
        utils.copy_files_and_create_dirs(files)

    # Prepare data;
    if config['model_batch_size'] < config['batch_size']:
        config['model_batch_size'] = config['batch_size']
    M_batch_size = config['model_batch_size']
    loaders = utils.get_data_loaders(**{**config, 'batch_size': M_batch_size,
                                        'start_itr': state_dict['itr']})

    # Loaders are loaded, prepare the training function
    train = model.training_function(M, M_par, state_dict, config, loss_func=loss)
    # Prepare Sample function for use with inception metrics

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])
        M.train()
        for i, x in enumerate(pbar):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure M is in training mode, just in case they got set to eval
            if config['M_fp16']:
                for key in x.keys():
                    x[key] = x[key].to(device).half()
            else:
                for key in x.keys():
                    x[key] = x[key].to(device)
            y = x['target']

            metrics = train(x, y)
            train_log.log(itr=int(state_dict['itr']), **metrics)

            # If using my progbar, print metrics.
            if config['pbar'] == 'mine':
                print(', '.join(['epoch: %d' % state_dict['epoch']]
                                + ['%s : %+4.3f' % (key, metrics[key])
                                   for key in metrics]), end=' ')
            # break

        # step learning rate decay
        if M.adjust_learning_rate:
            M.adjust_learning_rate(state_dict['epoch'])

        # Save weights and copies as configured at specified interval
        if not (epoch % config['save_every']):
            if config['M_eval_mode']:
                print('Switching M to eval mode...')
                M.eval()
            model.save_and_sample(M_par, M, state_dict, config, experiment_name)


        # Test every specified interval
        if not (epoch % config['test_every']):
            if config['M_eval_mode']:
                print('Switching M to eval mode...')
                M.eval()
            model.test(M_par, M, loaders[1], state_dict, config, experiment_name, test_log, device=device)
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()