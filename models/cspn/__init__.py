import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
import collections
import math

import losses
import os
import time
import metrics
import utils
from utils import colored_depthmap_tensor
from .torch_resnet_cspn_nyu import resnet18

class model(nn.Module):
    def __init__(self, scales=4, base_width=32, dec_img=True, colorize=False, M_lr=2e-4, M_B1=0.0, M_B2=0.999, adam_eps=1e-8,
                 M_mixed_precision=False, M_fp16=False, layers=18, pretrained=True, inchannel=1, **kwargs):
        super(model, self).__init__()
        output_size = kwargs['resolution']        

        cspn_config = {'step': 24, 'norm_type': "8sum"}

        self.net = resnet18(pretrained=pretrained, cspn_config=cspn_config)

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = M_lr, M_B1, M_B2, adam_eps
        if M_mixed_precision:
            print('Using fp16 adam in D...')
            self.optim = utils.Adam16(params=self.net.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=1e-4, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.net.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=1e-4, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = optim.lr_scheduler.ExponentialLR(self.optim, gamma=100) # if self.progressive else {}

        self.m = nn.MaxPool2d(2, stride=2)

    def forward(self, inputs):
        rgb = F.interpolate(inputs[:, :3], size=(352, 1216), mode='bilinear')
        sparse_depth = F.interpolate(self.m(inputs[:, 3:]), size=(352, 1216), mode='nearest')

        output = self.net(torch.cat((rgb, sparse_depth), dim=1))

        output = F.interpolate(output, size=(1200, 1920), mode='bilinear')
        return output

    def adjust_learning_rate(self, epoch):
         lr = self.lr * 0.5 ** (epoch // 5)
         for param_group in self.optim.param_groups:
             param_group['lr'] = lr


# Parallelized V to minimize cross-gpu communication
class parallel(nn.Module):
    def __init__(self, DC):
        super(parallel, self).__init__()
        self.DC = DC

    def forward(self, sparse_depth, train=False):
        # If training V, enable grad tape
        with torch.set_grad_enabled(train):
            # Get output
            out = self.DC(sparse_depth)
            # Cast as necessary
            # By now we don't use half precision
            # if self.V_DC.fp16:
            #     out = [i.half() for i in out]
            return out

def training_function(M, M_para, state_dict, config, loss_func=losses.MaskedMSELoss()):
    def train(dic, y):
        x = dic['inputs']

        # How many chunks to split x and y into?
        _x = torch.split(x, config['batch_size'])
        _y = torch.split(y, config['batch_size'])

        if config['toggle_grads']:
            utils.toggle_grad(M, True)

        M.optim.zero_grad()
        for j in range(len(_x)):
            inputs = _x[j]
            __y = _y[j]

            gen_depth = M_para(inputs, train=True)
            loss = loss_func(gen_depth, __y)

            loss.backward()
        M.optim.step()

        out = {'Loss': float(loss.item())}
        # Return G's loss and the components of D's loss.
        return out

    return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(M_par, M, state_dict, config, experiment_name):
    utils.save_weights(M, state_dict, config['weights_root'],
                       experiment_name, None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(M, state_dict, config['weights_root'],
                           experiment_name, 'copy%d' %  state_dict['save_num'])
        state_dict['save_num'] = (state_dict['save_num'] + 1) % config['num_save_copies']



def test(M_par, M, loaders, state_dict, config, experiment_name, test_log, device):
    results = metrics.AverageMeter()
    test_sample_num = min(config['batch_size'], config['num_test_samples'])
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
        pbar = utils.progress(loaders, displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
        pbar = tqdm(loaders)
    if config['toggle_grads']:
        utils.toggle_grad(M, False)
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

        # How many chunks to split x and y into?
        _x = torch.split(x, config['batch_size'])
        _y = torch.split(y, config['batch_size'])

        data_time = time.time() - date_time_start
        for j in range(len(_x)):
            inputs = _x[j]
            __y = _y[j]

            gpu_time_start = time.time()
            with torch.no_grad():
                gen_depth = M_par(inputs)
            gpu_time = time.time() - gpu_time_start

            r = metrics.Result()
            r.evaluate(gen_depth, __y)
            results.update(r, gpu_time, data_time, gen_depth.size(0))

        avg_result = results.average()

        date_time_start = time.time()
        # break

        # If using my progbar, print metrics.
        if config['pbar'] == 'mine':
            print(', '.join(['itr: %d' % state_dict['itr']]
                            + ['%s : %+4.3f, %s : %+4.3f' % 
                            ('mae', avg_result.mae,
                             'rmse', avg_result.rmse
                             )]), end=' ')

    mae_loss = avg_result.mae
    rmse_loss = avg_result.rmse
    if (rmse_loss < state_dict['best_loss']):
        print('%s improved over previous best, saving checkpoint...' % 'rmse loss')
        utils.save_weights(M, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'])
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1) % config['num_best_copies']
        state_dict['best_loss'] = rmse_loss

    # Log results to file
    test_log.log(itr=int(state_dict['epoch']), 
                irmse = float(avg_result.irmse),
                imae = float(avg_result.imae),
                mse = float(avg_result.mse),
                rmse = float(avg_result.rmse),
                mae = float(avg_result.mae),
                absrel = float(avg_result.absrel),
                squared_rel = float(avg_result.squared_rel),
                lg10 = float(avg_result.lg10),
                delta1 = float(avg_result.delta1),
                delta2 = float(avg_result.delta2),
                delta3 = float(avg_result.delta3),
                data_time = float(avg_result.data_time),
                gpu_time = float(avg_result.gpu_time),
                silog = float(avg_result.silog)
                )

    # save output of sample
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_test_samples%d.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['epoch'])
    torchvision.utils.save_image(utils.colored_depthmap_tensor(gen_depth).float().cpu(), image_filename,
                                 nrow=int(gen_depth.shape[0] ** 0.5), normalize=False)

    test_target_filename = '%s/%s/fixed_test_samples_target.jpg' % (config['samples_root'],
                                                    experiment_name)
    torchvision.utils.save_image(utils.colored_depthmap_tensor(__y).float().cpu(), test_target_filename,
                              nrow=int(__y.shape[0] ** 0.5), normalize=False)
