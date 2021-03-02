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

class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        ### input

        x0 = self.init(input)
        if pre_x4 is not None:
            x0 = x0 + F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)

        return x0, x1, x2


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):
        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return x0, x1, x2, x3, x4


class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   nn.Conv2d(layers // 2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        ### prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

    def forward(self, inputs, train=False):
        input_d = inputs[:, 3:]
        input_rgb = inputs[:, :3]

        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        output_d11 = F.interpolate(output_d11, size=(1200, 1920), mode='bilinear')
        output_d12 = F.interpolate(output_d12, size=(1200, 1920), mode='bilinear')
        output_d14 = F.interpolate(output_d14, size=(1200, 1920), mode='bilinear')

        if not train:
            return output_d11
        return (output_d11, output_d12, output_d14)


class model(nn.Module):
    def __init__(self, scales=4, base_width=32, dec_img=True, colorize=False, M_lr=2e-4, M_B1=0.0, M_B2=0.999, adam_eps=1e-8,
                 M_mixed_precision=False, M_fp16=False, layers=18, pretrained=True, inchannel=1, **kwargs):
        super(model, self).__init__()
        output_size = kwargs['resolution']        
        self.net = network()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = M_lr, M_B1, M_B2, adam_eps
        if M_mixed_precision:
            print('Using fp16 adam in D...')
            self.optim = utils.Adam16(params=self.net.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.net.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = optim.lr_scheduler.ExponentialLR(self.optim, gamma=100) # if self.progressive else {}

        self.m = nn.MaxPool2d(2, stride=2)

    def forward(self, inputs, train=False):
        rgb = F.interpolate(inputs[:, :3], size=(512, 768), mode='bilinear')
        sparse_depth = F.interpolate(self.m(inputs[:, 3:]), size=(512, 768), mode='nearest')

        output = self.net(torch.cat((rgb, sparse_depth), dim=1), train=train)

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
            out = self.DC(sparse_depth, train=train)
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

            outputs = M_para(inputs, train=True)

            # Calculate loss for valid pixel in the ground truth
            loss11 = loss_func(outputs[0], __y)
            loss12 = loss_func(outputs[1], __y)
            loss14 = loss_func(outputs[2], __y)

            if state_dict['epoch'] < 6:
                loss = loss14 + loss12 + loss11
            elif state_dict['epoch'] < 35:
                loss = 0.1 * loss14 + 0.1 * loss12 + loss11
            else:
                loss = loss11

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

