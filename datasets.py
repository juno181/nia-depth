''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import os
import os.path
import sys
import random
from PIL import Image
import numpy as np
import math
from tqdm import tqdm, trange
import string
import random

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
TXT_EXTENSIONS = ['.txt']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def is_text_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known text extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in TXT_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_kitti_dataset(file_root):
    kitti_data = []
    # os.path.expanduser(root)
    for dir_name in tqdm(sorted(os.listdir(file_root))):
        d = os.path.join(file_root, dir_name)
        if not os.path.isdir(d):
            continue
        dd = os.path.join(d, 'proj_depth/groundtruth/image_02')
        for root, _, fnames in sorted(os.walk(dd)):
            fnames_with_img = [i for i in sorted(fnames)]
            num_imgs = len(fnames_with_img)

            for idx, fname in enumerate(sorted(fnames_with_img)):
                if os.path.exists(os.path.join(os.path.join(d, 'image_02/data'), fname)) and \
                        os.path.exists(os.path.join(os.path.join(d, 'proj_depth/velodyne_raw/image_02'), fname)) and \
                        os.path.exists(os.path.join(os.path.join(d, 'oxts/data'), fname.replace('png', 'txt'))):
                    path_depth = os.path.join(root, fname)
                    path_img = os.path.join(os.path.join(d, 'image_02/data'), fname)
                    path_sparsedepth = os.path.join(os.path.join(d, 'proj_depth/velodyne_raw/image_02'), fname)
                    path_oxts = os.path.join(os.path.join(d, 'oxts/data'), fname.replace('png', 'txt'))
                    item = (path_img, path_depth, path_sparsedepth, path_oxts)
                    kitti_data.append(item)
                else:
                    print('missing image!', d, fname)

    return kitti_data


def make_nia_dataset(file_root):
    nia_data = []
    # os.path.expanduser(root)
    depth_dir = 'depth'
    rgb_dir = 'image_0'
    dafualt_fname = "%s"

    for dir_name in tqdm(sorted(os.listdir(file_root))):
        d = os.path.join(file_root, dir_name)

        if not os.path.isdir(d):
            continue

        dd = os.path.join(d, depth_dir)
        for fname in sorted(os.listdir(dd)):
            if os.path.exists(os.path.join(os.path.join(d, rgb_dir), fname + '.jpg')) and \
                os.path.join(os.path.join(d, depth_dir), fname + "/0.png"):
                path_rgb = os.path.join(os.path.join(os.path.join(d, rgb_dir), fname + '.jpg'))
                path_depth = os.path.join(os.path.join(d, depth_dir), fname + "/0.png")
                item = (path_rgb, path_depth)
                nia_data.append(item)
            else:
                print('missing image!', fname)

    return nia_data


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def pil_depth_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.fromarray(np.array(Image.open(f)).astype("float32"))
        return img.convert('F')


def default_depth_loader(path):
    from torchvision import get_image_backend
    return pil_depth_loader(path)


def get_kitti_h5pylen(root, hdf5_name):
    count = 0
    for root, _, fnames in sorted(os.walk(root)):
        for names in fnames:
            if hdf5_name in names:
                count += 1
    return count


##################################################################################################################
class ImageFolder_Kitti(data.Dataset):
    """A Kitti data loader where the images are arranged in this way: ::

        root/2011_09_26_drive_0002_sync/images/xxx.png
        root/2011_09_26_drive_0002_sync/depth/xxx.png
        root/2011_09_26_drive_0002_sync/oxts/xxx.png
        root/2011_09_26_drive_0002_sync/images/xxy.png
        root/2011_09_26_drive_0002_sync/images/xxz.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, depth_transform=None, target_transform=None,
                 loader=default_loader, depth_loader=default_depth_loader, load_in_mem=False, augment=False, mode='train',
                 num_of_skip_img=10, n_sample=500,
                 index_filename='kitti_imgs.npz', seq_path='', **kwargs):
        # Load pre-computed image directory walk
        if os.path.exists(index_filename):
            print('Loading pre-saved Index file %s...' % index_filename)
            data_path = np.load(index_filename, allow_pickle=True)['data']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating  Index file %s...' % index_filename)
            data_path = make_kitti_dataset(root)
            np.savez_compressed(index_filename, **{'data': data_path})
        if len(data_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                   "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.transform = transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        self.loader = loader
        self.depth_loader = depth_loader
        self.load_in_mem = load_in_mem
        self.augment = augment
        self.image_size = kwargs['image_size']
        self.mode = mode
        # self.generator = torch.Generator()
        self.n_sample = n_sample

        # self.extrinsic = get_extrinsic_file(self.extrinsic_path, self.scene_num)
        # self.file_length = np.array([len(x) for x in self.extrinsic])

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data = []
            seq = []
            for i in range(len(data)):
                img_path, depth_path, sparsedepth_path, _ = data_path[i][0], data_path[i][1], \
                                                            data_path[i][2], \
                                                            data_path[i][3]
                imgs = self.loader(img_path)
                depth = self.depth_loader(depth_path)
                sparsedepth = self.depth_loader(sparsedepth_path)
                # oxts = txt_loader(oxts_path)
                self.data.append((imgs, depth, sparsedepth))
        else:
            self.data = data_path

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img = self.data[index]
            depth = self.data[index]
            sparsedepth = self.data[index]
        else:
            img_path = os.path.join(self.root, self.data[index][0])
            depth_path = os.path.join(self.root, self.data[index][1])
            sparsedepth_path = os.path.join(self.root, self.data[index][2])
            img = self.loader(img_path)
            depth = self.depth_loader(depth_path)
            sparsedepth = self.depth_loader(sparsedepth_path)

        if (self.mode=='train') & self.augment:
            _scale = random.uniform(1.0, 1.5)
            scale = int(self.image_size[0] * _scale)
            degree = random.uniform(-5.0, 5.0)
            flip = random.uniform(0.0, 1.0)
        
            # Horizontal flip
            if flip > 0.5:
                img = F.hflip(img)
                depth = F.hflip(depth)
                sparsedepth = F.hflip(sparsedepth)

            # Rotation
            img = F.rotate(img, angle=degree, resample=Image.BICUBIC)
            depth = F.rotate(depth, angle=degree, resample=Image.NEAREST)
            sparsedepth = F.rotate(sparsedepth, angle=degree, resample=Image.NEAREST)

            # Color jitter
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.6, 1.4)

            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)

            # Resize
            img = F.resize(img, scale, Image.BICUBIC)
            depth = F.resize(depth, scale, Image.NEAREST)
            sparsedepth = F.resize(sparsedepth, scale, Image.NEAREST)
                               
            img = F.center_crop(img, self.image_size)
            depth = F.center_crop(depth, self.image_size)
            sparsedepth = F.center_crop(sparsedepth, self.image_size)

            img = self.transform(img)
            depth = self.depth_transform(depth)
            sparsedepth = self.depth_transform(sparsedepth)
           
            depth = depth / _scale
            sparsedepth = sparsedepth / _scale

        elif self.transform is not None:
            img = self.transform(img)
            depth = self.depth_transform(depth)
            sparsedepth = self.depth_transform(sparsedepth)
            
        inputs = torch.cat((img, sparsedepth / 256.), dim=0)
        dat = {
            'target': depth / 256.,
            'inputs': inputs
        }

        return dat

    def __len__(self):
        return len(self.data)

    def get_sparse_depth(self, depth, num_sample):
        channel, height, width = depth.shape

        assert channel == 1

        idx_nnz = torch.nonzero(depth.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = depth * mask.type_as(depth)

        return dep_sp

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


##################################################################################################################
class ImageFolder_nia(data.Dataset):
    """A nipa data loader where the images are arranged in this way: ::

        root/images/xxx.png
        root/depth/xxx.png
        root/gt/xxx.png
        root/images/xxy.png
        root/images/xxz.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, depth_transform=None, target_transform=None,
                 loader=default_loader, depth_loader=default_depth_loader, load_in_mem=False, augment=False, mode='train',
                 num_of_skip_img=10, n_sample=500,
                 index_filename='kitti_imgs.npz', seq_path='', **kwargs):
        # Load pre-computed image directory walk
        if os.path.exists(index_filename):
            print('Loading pre-saved Index file %s...' % index_filename)
            data_path = np.load(index_filename, allow_pickle=True)['data']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating  Index file %s...' % index_filename)
            data_path = make_nia_dataset(root)
            np.savez_compressed(index_filename, **{'data': data_path})
        if len(data_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                   "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.transform = transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        self.loader = loader
        self.depth_loader = depth_loader
        self.load_in_mem = load_in_mem
        self.augment = augment
        self.image_size = kwargs['image_size']
        self.mode = mode
        # self.generator = torch.Generator()
        self.n_sample = n_sample

        # self.extrinsic = get_extrinsic_file(self.extrinsic_path, self.scene_num)
        # self.file_length = np.array([len(x) for x in self.extrinsic])

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data = []
            seq = []
            for i in range(len(data)):
                img_path, depth_path = data_path[i][0], data_path[i][1]
                imgs = self.loader(img_path)
                depth = self.depth_loader(depth_path)
                # oxts = txt_loader(oxts_path)
                self.data.append((imgs, depth))
        else:
            self.data = data_path

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img = self.data[index][0]
            depth = self.data[index][2]
        else:
            img_path = os.path.join(self.root, self.data[index][0])
            depth_path = os.path.join(self.root, self.data[index][1])
            img = self.loader(img_path)
            depth = self.depth_loader(depth_path)

        if (self.mode=='train') & self.augment:
            _scale = random.uniform(1.0, 1.5)
            scale = int(self.image_size[0] * _scale)
            degree = random.uniform(-5.0, 5.0)
            flip = random.uniform(0.0, 1.0)
        
            # Horizontal flip
            if flip > 0.5:
                img = F.hflip(img)
                depth = F.hflip(depth)

            # Rotation
            img = F.rotate(img, angle=degree, resample=Image.BICUBIC)
            depth = F.rotate(depth, angle=degree, resample=Image.NEAREST)

            # Color jitter
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.6, 1.4)

            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)

            # Resize
            img = F.resize(img, scale, Image.BICUBIC)
            depth = F.resize(depth, scale, Image.NEAREST)
                               
            img = F.center_crop(img, self.image_size)
            depth = F.center_crop(depth, self.image_size)

            img = self.transform(img)
            depth = self.depth_transform(depth)
           
            depth = depth / _scale

        elif self.transform is not None:
            img = self.transform(img)
            depth = self.depth_transform(depth)

        sparsedepth = self.get_sparse_depth(depth, self.n_sample)
        
        inputs = torch.cat((img, depth / 256.), dim=0)
        dat = {
            'target': depth / 256.,
            'inputs': inputs
        }

        return dat

    def __len__(self):
        return len(self.data)

    def get_sparse_depth(self, depth, num_sample):
        channel, height, width = depth.shape

        assert channel == 1

        idx_nnz = torch.nonzero(depth.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_idx//3]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = depth * mask.type_as(depth)

        return dep_sp

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

