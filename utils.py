#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
This file contains utility functions for bookkeeping, logging, and data loading.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

import datasets as dset

cmap = plt.cm.viridis

def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='I128_hdf5',
        help='Which Dataset to train on, out of I128, I256, C10, C100;'
             'Append "_hdf5" to use the hdf5 version for ISLVRC '
             '(default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Augment with random crops and flips (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers; consider using less for HDF5 '
             '(default: %(default)s)')
    parser.add_argument(
        '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
        help='Pin data into memory through dataloader? (default: %(default)s)')
    parser.add_argument(
        '--shuffle', action='store_true', default=False,
        help='Shuffle the data (strongly recommended)? (default: %(default)s)')
    parser.add_argument(
        '--load_in_mem', action='store_true', default=False,
        help='Load all data into memory? (default: %(default)s)')
    parser.add_argument(
        '--use_multiepoch_sampler', action='store_true', default=False,
        help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
    parser.add_argument(
        '--num_of_skip_img', type=int, default=20,
        help='How many images to skip frames? (default: %(default)s)')
    parser.add_argument(
        '--n_sample', type=int, default=500,
        help='How many points in sparse depth? (default: %(default)s)')

    ### Model stuff ###
    parser.add_argument(
        '--model', type=str, default='Zhou',
        help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--M_ch', type=int, default=64,
        help='Channel multiplier for M (default: %(default)s)')
    parser.add_argument(
        '--M_nl', type=str, default='relu',
        help='Activation function for M (default: %(default)s)')
    parser.add_argument(
        '--M_attn', type=str, default='64',
        help='What resolutions to use attention on for M (underscore separated) '
             '(default: %(default)s)')
    parser.add_argument(
        '--train_V', action='store_true', default=False,
        help='Train view syn model? (default: %(default)s)')

    ### Model init stuff ###
    parser.add_argument(
        '--seed', type=int, default=2147483647,
        help='Random seed to use; affects both initialization and '
             ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--M_init', type=str, default='kaiming',
        help='Init style to use for M(default: %(default)s)')
    parser.add_argument(
        '--skip_init', action='store_true', default=False,
        help='Skip initialization, ideal for testing when ortho init was used '
             '(default: %(default)s)')
    parser.add_argument(
        '--pretrained_experiment_name', type=str, default='',
        help='The name of experiment name where view syn model is loaded. '
             '(default: %(default)s)')

    ### S2D Model init stuff ###
    parser.add_argument(
        '--res_layers', type=int, default=50,
        help='number of resnet layers in S2D(default: %(default)s)')
    parser.add_argument(
        '--decoder_type', type=str, default='upproj',
        help='Upsampling layer type of S2D(default: %(default)s)')
    parser.add_argument(
        '--S2D_pretrained', action='store_false', default=True,
        help='Using pretrained model in S2D(default: %(default)s)')

    ### Optimizer stuff ###
    parser.add_argument(
        '--M_lr', type=float, default=5e-5,
        help='Learning rate to use for model (default: %(default)s)')
    parser.add_argument(
        '--M_B1', type=float, default=0.0,
        help='Beta1 to use for model (default: %(default)s)')
    parser.add_argument(
        '--M_B2', type=float, default=0.999,
        help='Beta2 to use for model (default: %(default)s)')

    ### Batch size, parallel, and precision stuff ###
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--model_batch_size', type=int, default=64,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--num_epochs', type=int, default=2000,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')
    parser.add_argument(
        '--M_fp16', action='store_true', default=False,
        help='Train with half-precision in M? (default: %(default)s)')
    parser.add_argument(
        '--M_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in M? '
             '(default: %(default)s)')

    ### Bookkeping stuff ###
    parser.add_argument(
        '--M_eval_mode', action='store_true', default=False,
        help='Run M in eval mode (running/standing stats?) at sample/test time? '
             '(default: %(default)s)')
    parser.add_argument(
        '--save_every', type=int, default=2000,
        help='Save every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_save_copies', type=int, default=2,
        help='How many copies to save (default: %(default)s)')
    parser.add_argument(
        '--num_test_samples', type=int, default=10,
        help='How many test sample images to save (less than batch size) (default: %(default)s)')
    parser.add_argument(
        '--num_best_copies', type=int, default=2,
        help='How many previous best checkpoints to save (default: %(default)s)')
    parser.add_argument(
        '--which_best', type=str, default='IS',
        help='Which metric to use to determine when to save new "best"'
             'checkpoints, one of IS or FID (default: %(default)s)')
    parser.add_argument(
        '--no_fid', action='store_true', default=False,
        help='Calculate IS only, not FID? (default: %(default)s)')
    parser.add_argument(
        '--test_every', type=int, default=5000,
        help='Test every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_inception_images', type=int, default=50000,
        help='Number of samples to compute inception metrics with '
             '(default: %(default)s)')
    parser.add_argument(
        '--hashname', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
             '(default: %(default)s)')
    parser.add_argument(
        '--base_root', type=str, default='',
        help='Default location to store all weights, samples, data, and logs '
             ' (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
    parser.add_argument(
        '--logs_root', type=str, default='logs',
        help='Default location to store logs (default: %(default)s)')
    parser.add_argument(
        '--tensorboard_root', type=str, default='tf_logs',
        help='Default location to store tensorboard logs (default: %(default)s)')
    parser.add_argument(
        '--samples_root', type=str, default='samples',
        help='Default location to store samples (default: %(default)s)')
    parser.add_argument(
        '--pbar', type=str, default='mine',
        help='Type of progressbar to use; one of "mine" or "tqdm" '
             '(default: %(default)s)')
    parser.add_argument(
        '--name_suffix', type=str, default='',
        help='Suffix for experiment name for loading weights for sampling '
             '(consider "best0") (default: %(default)s)')
    parser.add_argument(
        '--experiment_name', type=str, default='',
        help='Optionally override the automatic experiment naming with this arg. '
             '(default: %(default)s)')
    parser.add_argument(
        '--config_from_name', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
             '(default: %(default)s)')

    ### Numerical precision and SV stuff ###
    parser.add_argument(
        '--adam_eps', type=float, default=1e-8,
        help='epsilon value to use for Adam (default: %(default)s)')
    parser.add_argument(
        '--BN_eps', type=float, default=1e-5,
        help='epsilon value to use for BatchNorm (default: %(default)s)')

    ### Ortho reg stuff ###
    parser.add_argument(
        '--toggle_grads', action='store_true', default=True,
        help='Toggle M''s "requires_grad" settings when not training them? '
             ' (default: %(default)s)')

    ### Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='',
        help='Suffix for which weights to load (e.g. best0, copy0) '
             '(default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume training? (default: %(default)s)')
    parser.add_argument(
        '--resume_experiment_name', type=str, default='',
        help='experiment name for which weights to load '
             '(default: %(default)s)')

    ### Log stuff ###
    parser.add_argument(
        '--logstyle', type=str, default='%3.3e',
        help='What style to use when logging training metrics?'
             'One of: %#.#f/ %#.#e (float/exp, text),'
             'pickle (python pickle),'
             'npz (numpy zip),'
             'mat (MATLAB .mat file) (default: %(default)s)')
    parser.add_argument(
        '--log_D_spectra', action='store_true', default=False,
        help='Log the top 3 singular values in each SN layer in D? '
             '(default: %(default)s)')
    parser.add_argument(
        '--sv_log_interval', type=int, default=10,
        help='Iteration interval for logging singular values '
             ' (default: %(default)s)')
    parser.add_argument(
        '--save_code', action='store_true', default=False,
        help='Backup codes when running '
             '(default: %(default)s)')
    parser.add_argument(
        '--use_tensorboard', action='store_true', default=False,
        help='Save logs on tensorboard '
             '(default: %(default)s)')

    return parser


# Arguments for sample.py; not presently used in train.py
def add_sample_parser(parser):
    # parser.add_argument(
        # '--evaluate', action='store_true', default=False,
        # help='Evaluate images')
    parser.add_argument(
        '--predict_data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    return parser


# Convenience dicts
dset_dict = {'Kitti_train': dset.ImageFolder_Kitti,
             'Kitti_val': dset.ImageFolder_Kitti,
             'Kitti_test': dset.ImageFolder_Kitti,
             'nia_train': dset.ImageFolder_nia,
             'nia_val': dset.ImageFolder_nia,
             'nia_test': dset.ImageFolder_nia}
imsize_dict = {# 'Kitti_train': 224,
               # 'Kitti_test': 224,
               'Kitti_train': (352, 1216),
               'Kitti_val': (352, 1216),
               'Kitti_test': (352, 1216),
               'nia_train': (1200, 1920),
               'nia_val': (1200, 1920),
               'nia_test': (1200, 1920)
            #    'nia_train': (352, 1216),
            #    'nia_val': (352, 1216),
            #    'nia_test': (352, 1216)
               }
root_dict = {'Kitti_train': 'train',
             'Kitti_val': 'val',
             'Kitti_test': 'test',
             'nia_train': 'train',
             'nia_val': 'val',
             'nia_test': 'val'}
nclass_dict = {'Kitti_train': 21, 'Kitti_val': 21, 'Kitti_test': 21,
               'K224_train_hdf5': 21, 'K224_test_hdf5': 21}
# Number of classes to put per sample sheet               
classes_per_sheet_dict = {'Kitti_train': 21, 'Kitti_test': 21, 
                          'K224_train_hdf5': 21, 'K224_test_hdf5': 21}
activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True), }



# multi-epoch Dataset sampler to avoid memory leakage and enable resumption of
# training from the same sample regardless of if we stop mid-epoch
class MultiEpochSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly over multiple epochs

    Arguments:
        data_source (Dataset): dataset to sample from
        num_epochs (int) : Number of times to loop over the dataset
        start_itr (int) : which iteration to begin from
    """

    def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.num_epochs = num_epochs
        self.start_itr = start_itr
        self.batch_size = batch_size

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self):
        n = len(self.data_source)
        # Determine number of epochs
        num_epochs = int(np.ceil((n * self.num_epochs
                                  - (self.start_itr * self.batch_size)) / float(n)))
        # Sample all the indices, and then grab the last num_epochs index sets;
        # This ensures if we're starting at epoch 4, we're still grabbing epoch 4's
        # indices
        out = [torch.randperm(n) for epoch in range(self.num_epochs)][-num_epochs:]
        # Ignore the first start_itr % n indices of the first epoch
        out[0] = out[0][(self.start_itr * self.batch_size % n):]
        # if self.replacement:
        # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # return iter(.tolist())
        output = torch.cat(out).tolist()
        print('Length dataset output is %d' % len(output))
        return iter(output)

    def __len__(self):
        return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size


# Convenience function to centralize all data loaders
def get_data_loaders(dataset, data_root=None, augment=False, batch_size=64,
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     **kwargs):
    # Append /FILENAME.hdf5 to root if using hdf5
    # data_root += '/%s' % root_dict[dataset]
    print('Using dataset root location %s' % data_root)

    which_dataset = dset_dict[dataset]
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    image_size = imsize_dict[dataset]
    # For image folder datasets, name of the file where we store the precomputed
    # image locations to avoid having to walk the dirs every time we load.
    dataset_kwargs = {'index_filename': '%s_imgs.npz' % dataset, 'hdf5_name': root_dict[dataset],
                      'n_sample': kwargs['n_sample'], 'image_size': image_size}
    dataset_kwargs_val = {'index_filename': '%s_imgs.npz' % dataset.replace('train', 'val'),
                           'hdf5_name': root_dict[dataset.replace('train', 'val')],
                           'n_sample': kwargs['n_sample'], 'train': False, 'image_size': image_size}
    #dataset_kwargs_test = {'index_filename': '%s_imgs.npz' % dataset.replace('train', 'test'),
    #                       'hdf5_name': root_dict[dataset.replace('train', 'test')],
    #                       'n_sample': kwargs['n_sample'], 'train': False}

    # HDF5 datasets have their own inbuilt transform, no need to train_transform
    if 'hdf5' in dataset:
        train_transform = None
        test_transform = None
    else:
        if augment:
            print('Data will be augmented...')
            if dataset in ['Kitti_train', 'Kitti_test']:
                train_transform = []
            else:
                train_transform = []
        else:
            print('Data will not be augmented...')
            train_transform = []
            # train_transform = [transforms.Resize(image_size), transforms.CenterCrop]
        train_depth_transform = transforms.Compose(train_transform + [
            transforms.CenterCrop(image_size),
            # transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor()])
        train_transform = transforms.Compose(train_transform + [
            transforms.CenterCrop(image_size),
            # transforms.Resize(image_size, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        test_transform = transforms.Compose([transforms.CenterCrop(image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(norm_mean, norm_std)])
    train_set = which_dataset(root=data_root + '/%s' % root_dict[dataset], transform=train_transform, depth_transform=train_depth_transform,
                              load_in_mem=load_in_mem, augment=augment, mode='train', **dataset_kwargs)
    val_set = which_dataset(root=data_root + '/%s' % root_dict[dataset.replace('train', 'val')], transform=test_transform, depth_transform=train_depth_transform,
                             load_in_mem=load_in_mem, augment=augment, mode='val', **dataset_kwargs_val)
    # test_set = which_dataset(root=data_root.replace('train', 'test'), transform=test_transform, depth_transform=train_depth_transform,
    #                          load_in_mem=load_in_mem, **dataset_kwargs_test)

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)

    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(test_set, num_epochs, start_itr, batch_size)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                 sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                 shuffle=False, **loader_kwargs)
    loaders.append(val_loader)

    # if use_multiepoch_sampler:
        # print('Using multiepoch sampler from start_itr %d...' % start_itr)
        # loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        # sampler = MultiEpochSampler(test_set, num_epochs, start_itr, batch_size)
        # test_loader = DataLoader(test_set, batch_size=batch_size,
                                #  sampler=sampler, **loader_kwargs)
    # else:
        # loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                        #  'drop_last': drop_last}  # Default, drop last incomplete batch
        # test_loader = DataLoader(test_set, batch_size=batch_size,
                                #  shuffle=False, **loader_kwargs)
    # loaders.append(test_loader)
    return loaders


# Convenience function to centralize all data loaders
def get_test_data_loader(dataset, data_root=None, augment=False, batch_size=64,
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     **kwargs):
    # Append /FILENAME.hdf5 to root if using hdf5
    # data_root += '/%s' % root_dict[dataset]
    print('Using dataset root location %s' % data_root)

    which_dataset = dset_dict[dataset]
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    image_size = imsize_dict[dataset]
    # For image folder datasets, name of the file where we store the precomputed
    # image locations to avoid having to walk the dirs every time we load.
    dataset_kwargs = {'index_filename': '%s_imgs.npz' % dataset, 'hdf5_name': root_dict[dataset],
                      'n_sample': kwargs['n_sample'], 'image_size': image_size}
    dataset_kwargs_val = {'index_filename': '%s_imgs.npz' % dataset.replace('train', 'val'),
                           'hdf5_name': root_dict[dataset.replace('train', 'val')],
                           'n_sample': -1, 'train': False, 'image_size': image_size}
    dataset_kwargs_test = {'index_filename': '%s_imgs.npz' % dataset.replace('train', 'test'),
                          'hdf5_name': root_dict[dataset.replace('train', 'test')],
                          'n_sample': -1, 'train': False, 'image_size': image_size}

    # HDF5 datasets have their own inbuilt transform, no need to train_transform
    if 'hdf5' in dataset:
        train_transform = None
        test_transform = None
    else:
        # train_transform = [transforms.Resize(image_size), transforms.CenterCrop]
        train_depth_transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor()])
        train_transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        test_transform = transforms.Compose([transforms.CenterCrop(image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(norm_mean, norm_std)])
    train_set = which_dataset(root=data_root + '/%s' % root_dict[dataset], transform=train_transform, depth_transform=train_depth_transform,
                              load_in_mem=load_in_mem, augment=augment, mode='train', **dataset_kwargs)
    val_set = which_dataset(root=data_root + '/%s' % root_dict[dataset.replace('train', 'val')], transform=test_transform, depth_transform=train_depth_transform,
                             load_in_mem=load_in_mem, augment=augment, mode='val', **dataset_kwargs_val)
    test_set = which_dataset(root=data_root + '/%s' % root_dict[dataset.replace('train', 'test')], transform=test_transform, depth_transform=train_depth_transform,
                             load_in_mem=load_in_mem, **dataset_kwargs_test)

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)

    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(test_set, num_epochs, start_itr, batch_size)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                 sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                 shuffle=False, **loader_kwargs)
    loaders.append(val_loader)

    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(test_set, num_epochs, start_itr, batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False, **loader_kwargs)
    loaders.append(test_loader)
    return loaders


# Utility file to seed rngs
def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
    if config['base_root']:
        print('Pegging all root folders to base root %s' % config['base_root'])
        for key in ['data', 'weights', 'logs', 'samples']:
            config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
    return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
    for key in ['weights_root', 'logs_root', 'samples_root']:
        if not os.path.exists(config[key]):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key])


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes & not in blacklist
            if len(param.shape) < 2 or param in blacklist:
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 - torch.eye(w.shape[0], device=w.device), w))
            param.grad.data += strength * grad.view(param.shape)


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


# Function to join strings or ignore them
# Base string is the string to link "strings," while strings
# is a list of strings or Nones.
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


# Save a model's weights, optimizer, and the state_dict
def save_weights(M, state_dict, weights_root, experiment_name,
                 name_suffix=None):
    root = '/'.join([weights_root, experiment_name])
    if not os.path.exists(root):
        os.mkdir(root)
    if name_suffix:
        print('Saving weights to %s/%s...' % (root, name_suffix))
    else:
        print('Saving weights to %s...' % root)
    torch.save(M.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['M', name_suffix])))
    torch.save(M.optim.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['M_optim', name_suffix])))
    torch.save(state_dict,
               '%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))


# Load a model's weights, optimizer, and the state_dict
def load_weights(M, state_dict, weights_root, experiment_name, model_name='M',
                 name_suffix="copy0", strict=False, load_optim=True):
    root = '/'.join([weights_root, experiment_name])
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    if M is not None:
        M.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', [model_name, name_suffix]))),
            strict=strict)
        if load_optim:
            M.optim.load_state_dict(
                torch.load('%s/%s.pth' % (root, join_strings('_', [model_name + '_optim', name_suffix]))))
    # Load state dict
    for item in state_dict:
        state_dict[item] = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))[item]


''' MetricsLogger originally stolen from VoxNet source code.
    Used for logging inception metrics'''


class MetricsLogger(object):
    def __init__(self, fname, reinitialize=False):
        self.fname = fname
        self.reinitialize = reinitialize
        if os.path.exists(self.fname):
            if self.reinitialize:
                print('{} exists, deleting...'.format(self.fname))
                os.remove(self.fname)

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')


# Logstyle is either:
# '%#.#f' for floating point representation in text
# '%#.#e' for exponent representation in text
# 'npz' for output to npz # NOT YET SUPPORTED
# 'pickle' for output to a python pickle # NOT YET SUPPORTED
# 'mat' for output to a MATLAB .mat file # NOT YET SUPPORTED
class MyLogger(object):
    def __init__(self, fname, reinitialize=False, logstyle='%3.3f'):
        self.root = fname
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.reinitialize = reinitialize
        self.metrics = []
        self.logstyle = logstyle  # One of '%3.3f' or like '%3.3e'

    # Delete log if re-starting and log already exists
    def reinit(self, item):
        if os.path.exists('%s/%s.log' % (self.root, item)):
            if self.reinitialize:
                # Only print the removal mess
                if 'sv' in item:
                    if not any('sv' in item for item in self.metrics):
                        print('Deleting singular value logs...')
                else:
                    print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
                os.remove('%s/%s.log' % (self.root, item))

    # Log in plaintext; this is designed for being read in MATLAB(sorry not sorry)
    def log(self, itr, **kwargs):
        for arg in kwargs:
            if arg not in self.metrics:
                if self.reinitialize:
                    self.reinit(arg)
                self.metrics += [arg]
            if self.logstyle == 'pickle':
                print('Pickle not currently supported...')
                # with open('%s/%s.log' % (self.root, arg), 'a') as f:
                # pickle.dump(kwargs[arg], f)
            elif self.logstyle == 'mat':
                print('.mat logstyle not currently supported...')
            else:
                with open('%s/%s.log' % (self.root, arg), 'a') as f:
                    f.write('%d: %s\n' % (itr, self.logstyle % kwargs[arg]))


# Write some metadata to the logs directory
def write_metadata(logs_root, experiment_name, config, state_dict):
    with open(('%s/%s/metalog.txt' %
               (logs_root, experiment_name)), 'w') as writefile:
        writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
        writefile.write('config: %s\n' % str(config))
        writefile.write('state: %s\n' % str(state_dict))


"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
Andy's adds: time elapsed in addition to ETA, makes it possible to add
estimated time to 1k iters instead of estimated time to completion.
"""


def progress(items, desc='', total=None, min_delay=0.1, displaytype='s1k'):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                desc, n + 1, total, n / float(total) * 100), end=" ")
            if n > 0:

                if displaytype == 's1k':  # minutes/seconds for 1000 iters
                    next_1000 = n + (1000 - n % 1000)
                    t_done = t_now - t_start
                    t_1k = t_done / n * next_1000
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_1k - t_done, 60))
                    print("(TE/ET1k: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
                else:  # displaytype == 'eta':
                    t_done = t_now - t_start
                    t_total = t_done / n * total
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))
                    print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")

            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))


# Convenience debugging function to print out gradnorms and shape from each layer
# May need to rewrite this so we can actually see which parameter is which
def print_grad_norms(net):
    gradsums = [[float(torch.norm(param.grad).item()),
                 float(torch.norm(param).item()), param.shape]
                for param in net.parameters()]
    order = np.argsort([item[0] for item in gradsums])
    print(['%3.3e,%3.3e, %s' % (gradsums[item_index][0],
                                gradsums[item_index][1],
                                str(gradsums[item_index][2]))
           for item_index in order])


# Get singular values to log. This will use the state dict to find them
# and substitute underscores for dots.
def get_SVs(net, prefix):
    d = net.state_dict()
    return {('%s_%s' % (prefix, key)).replace('.', '_'):
                float(d[key].item())
            for key in d if 'sv' in key}


# Name an experiment based on its config
def name_from_config(config):
    name = '_'.join([
        item for item in [
            config['model'],
            config['name_suffix'] if config['name_suffix'] else None,
        ]
        if item is not None])
    # dogball
    if config['hashname']:
        return hashname(name)
    else:
        return name


from typing import List, Tuple
import fnmatch
import shutil


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> \
        List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


# A simple function to produce a unique experiment name from the animal hashes.
def hashname(name):
    h = hash(name)
    a = h % len(animal_hash.a)
    h = h // len(animal_hash.a)
    b = h % len(animal_hash.b)
    h = h // len(animal_hash.c)
    c = h % len(animal_hash.c)
    return animal_hash.a[a] + animal_hash.b[b] + animal_hash.c[c]


# Get GPU memory, -i is the index
def query_gpu(indices):
    os.system('nvidia-smi -i 0 --query-gpu=memory.free --format=csv')


# Convenience function to count the number of parameters in a module
def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))


class Adam16(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

    # Safety modification to make sure we floatify our state
    def load_state_dict(self, state_dict):
        super(Adam16, self).load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
                self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
                self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                    # Fp32 copy of the weights
                    state['fp32_p'] = p.data.float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], state['fp32_p'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
                p.data = state['fp32_p'].half()

        return loss

def depth_normalize(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())

def depth_abs_error_map(depth, gt):
    invalid_mask = gt < 0.001

    # convert from meters to mm
    output_mm = 1e3 * depth
    target_mm = 1e3 * gt

    abs_diff = (output_mm - target_mm).abs()
    abs_diff[invalid_mask] = 0

    # diff_map = depth_normalize(abs_diff)
    # maxRatio = torch.max(depth / gt, gt / depth)
    # delta1_map = depth
    # delta1_map[(delta1_map < 1.25)] = 0
    # delta1_map = depth_normalize(delta1_map)

    # return diff_map, delta1_map
    return abs_diff

def depth_mse_error_map(depth, gt):
    invalid_mask = gt < 0.001

    # convert from meters to mm
    output_mm = 1e3 * depth
    target_mm = 1e3 * gt

    abs_diff = (output_mm - target_mm).abs() ** 2
    abs_diff[invalid_mask] = 0

    # diff_map = depth_normalize(abs_diff)
    # maxRatio = torch.max(depth / gt, gt / depth)
    # delta1_map = depth
    # delta1_map[(delta1_map < 1.25)] = 0
    # delta1_map = depth_normalize(delta1_map)

    # return diff_map, delta1_map
    return abs_diff

def gray_error_map(img, gt):
    abs_diff = (img - gt).abs()
    diff_map = depth_normalize(abs_diff)

    return diff_map


def colored_depthmap_tensor(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = torch.min(depth).item()
    if d_max is None:
        d_max = torch.max(depth).item()
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_color = torch.from_numpy(cmap(depth_relative.cpu().numpy())).squeeze(1).permute((0, 3, 1, 2))[:, :3, :, :]
    return depth_color  # N, 3, H, W


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C