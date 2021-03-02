''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses
import tqdm
import time
import numpy as np
import metrics
from torch.nn import functional as F


def predict_function(V, state_dict, config):
    def predict(x):
        if config['toggle_grads']:
            utils.toggle_grad(V, False)

        with torch.no_grad():
            gen_depth = V(x)

        return gen_depth
    return predict


def evaluate_function(V, state_dict, config):
    def evaluate(loader, results, average_results, device):
        average_results.reset()
        # generate fixed output sample
        loader.dataset.generator = torch.Generator().manual_seed(config['seed'])
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(loader, displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loader)
            
            
        date_time_start = time.time()
        for i, dic in enumerate(pbar):
            if config['M_fp16']:
                for key in dic:
                    dic[key] = dic[key].to(device).half()
            else:
                for key in dic:
                    dic[key] = dic[key].to(device)

            x = dic['inputs']
            y = dic['target']
            data_time = time.time() - date_time_start

            if config['toggle_grads']:
                utils.toggle_grad(V, False)

            gpu_time_start = time.time()
            with torch.no_grad():
                gen_depth = V(x)

            gpu_time = time.time() - gpu_time_start

            r = metrics.Result()
            r.evaluate(gen_depth, y)
            average_results.update(r, gpu_time, data_time, gen_depth.size(0))
            
            avg_result = average_results.average()

            
            if i == 0:
                image_filename = config['model'] + 'inputs.jpg'
                torchvision.utils.save_image(x[:1, :3].float().cpu(), image_filename,
                                            nrow=int(x[:1].shape[0] ** 0.5), normalize=False)

                image_filename = config['model'] + 'inputs_depth.jpg'
                torchvision.utils.save_image(utils.colored_depthmap_tensor(x[:1, 3:]).float().cpu(), image_filename,
                                            nrow=int(gen_depth[:1].shape[0] ** 0.5), normalize=False)


                image_filename = config['model'] + '_samples.jpg'
                torchvision.utils.save_image(utils.colored_depthmap_tensor(gen_depth[:1]).float().cpu(), image_filename,
                                            nrow=int(gen_depth[:1].shape[0] ** 0.5), normalize=False)
                                            
                image_filename = config['model'] + 'gt.jpg'
                torchvision.utils.save_image(utils.colored_depthmap_tensor(y[:1]).float().cpu(), image_filename,
                                            nrow=int(y[:1].shape[0] ** 0.5), normalize=False)
                                            
                print('\n\nirmse: ', avg_result.irmse,
                    'imae: ', avg_result.imae,
                    'mse: ', avg_result.mse,
                    'rmse: ', avg_result.rmse,
                    'mae: ', avg_result.mae,
                    'absrel: ', avg_result.absrel,
                    'squared_rel: ', avg_result.squared_rel,
                    'lg10: ', avg_result.lg10,
                    'delta1: ', avg_result.delta1,
                    'delta2: ', avg_result.delta2,
                    'delta3: ', avg_result.delta3,
                    'data_time: ', avg_result.data_time,
                    'gpu_time: ', avg_result.gpu_time,
                    'silog: ', avg_result.silog,
                    '\n\n')
            

            print(', '.join(['itr: %d' % state_dict['itr']]
                        + ['%s : %+4.3f' % 
                        (
                        'rmse', avg_result.rmse
                        )]), end=' ')

        avg_result = average_results.average()

        date_time_start = time.time()

        return gen_depth, avg_result
    return evaluate
