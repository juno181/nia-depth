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

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x, skip4, skip3, skip2):
        x = self.layer1(x)
        x = self.layer2(torch.cat((x, skip2), dim=1))
        x = self.layer3(torch.cat((x, skip3), dim=1))
        x = self.layer4(torch.cat((x, skip4), dim=1))
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels, out_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(out_channels)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels, in_channels // 2)
        self.layer2 = convt(in_channels, in_channels // 2)
        self.layer3 = convt(in_channels // 2, in_channels // 2)
        self.layer4 = convt(in_channels // 4, in_channels // 2) 

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels, out_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels, out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels, in_channels // 2)
        self.layer2 = self.upconv_module(in_channels//1, in_channels // 4)
        self.layer3 = self.upconv_module(in_channels//2, in_channels // 8)
        self.layer4 = self.upconv_module(in_channels//4, in_channels // 4)

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels, out_channels):
            super(UpProj.UpProjModule, self).__init__()
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels, in_channels // 2)
        self.layer2 = self.UpProjModule(in_channels//1, in_channels // 4)
        self.layer3 = self.UpProjModule(in_channels//2, in_channels // 8)
        self.layer4 = self.UpProjModule(in_channels//4, in_channels // 4)

def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.decoder = choose_decoder(decoder, num_channels)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//4,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        skip4 = x
        x = self.layer2(x)
        skip3 = x
        x = self.layer3(x)
        skip2 = x
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x, skip4, skip3, skip2)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x


class model(nn.Module):
    def __init__(self, scales=4, base_width=32, dec_img=True, colorize=False, M_lr=2e-4, M_B1=0.0, M_B2=0.999, adam_eps=1e-8,
                 M_mixed_precision=False, M_fp16=False, layers=18, pretrained=True, inchannel=1, **kwargs):
        super(model, self).__init__()
        output_size = kwargs['resolution']        
        self.net = ResNet(34, "upproj", output_size, in_channels=4)

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

    def forward(self, inputs):
        rgb = F.interpolate(inputs[:, :3], size=(512, 768), mode='bilinear')
        sparse_depth = F.interpolate(self.m(inputs[:, 3:]), size=(512, 768), mode='nearest')

        output = self.net(torch.cat((rgb, sparse_depth), dim=1))

        output = F.interpolate(output, size=(1200, 1920), mode='bilinear')
        return output

    def adjust_learning_rate(self, epoch):
         lr = self.lr * 0.8 ** (epoch // 5)
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

