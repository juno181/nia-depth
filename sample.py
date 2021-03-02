''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange
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
import sample_fns
import metrics


def run(config):
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_loss': 999999, 'config': config}

    # Optionally, get the configuration from the state dict. This allows for
    # recovery of the config provided only a state dict and experiment name,
    # and can be convenient for writing less verbose sample shell scripts.
    if config['config_from_name']:
        utils.load_weights(None, None, state_dict, config['weights_root'],
                           config['experiment_name'], config['load_weights'], None,
                           strict=False, load_optim=False)
        # Ignore items which we might want to overwrite from the command line
        for item in state_dict['config']:
            if item not in ['base_root', 'batch_size']:
                config[item] = state_dict['config'][item]

    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    # config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    sys.path.insert(0, './models')
    model = __import__(config['model'])
    print('Import model from: ', config['model'])
    M = model.model(**config).to(device)
    utils.count_parameters(M)

    # Load weights
    print('Loading weights...')
    utils.load_weights(M, state_dict,
                       config['weights_root'], config['resume_experiment_name'])

    M_batch_size = config['batch_size']
    loaders = utils.get_test_data_loader(**{**config, 'batch_size': M_batch_size,
                                        'start_itr': state_dict['itr']})[2]

    if config['M_eval_mode']:
        print('Putting M in eval mode..')
        M.eval()
    else:
        print('M is in %s mode...' % ('training' if M.training else 'eval'))

    # inputs = get_input(config['predict_data_root'])
    # predict = sample_fns.predict_function(inputs)


    ############### evaluate results ###############
    # Sample function
    sample = sample_fns.evaluate_function(M, state_dict, config)

    results = metrics.Result()
    average_results = metrics.AverageMeter()

    sample_depth, average_results = sample(loaders, results, average_results, device)

    # Log results to file
    print('irmse: ', average_results.irmse,
          'imae: ', average_results.imae,
          'mse: ', average_results.mse,
          'rmse: ', average_results.rmse,
          'mae: ', average_results.mae,
          'absrel: ', average_results.absrel,
          'squared_rel: ', average_results.squared_rel,
          'lg10: ', average_results.lg10,
          'delta1: ', average_results.delta1,
          'delta2: ', average_results.delta2,
          'delta3: ', average_results.delta3,
          'data_time: ', average_results.data_time,
          'gpu_time: ', average_results.gpu_time,
          'silog: ', average_results.silog,
          )

    
    
    sample_metrics_fname = 'sample_%s_log.jsonl' %  config['model']
    sample_log = utils.MetricsLogger(sample_metrics_fname,
                                   reinitialize=True)

    # Log results to file
    sample_log.log(irmse = float(average_results.irmse),
                imae = float(average_results.imae),
                mse = float(average_results.mse),
                rmse = float(average_results.rmse),
                mae = float(average_results.mae),
                absrel = float(average_results.absrel),
                squared_rel = float(average_results.squared_rel),
                lg10 = float(average_results.lg10),
                delta1 = float(average_results.delta1),
                delta2 = float(average_results.delta2),
                delta3 = float(average_results.delta3),
                data_time = float(average_results.data_time),
                gpu_time = float(average_results.gpu_time),
                silog = float(average_results.silog)
                )


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
