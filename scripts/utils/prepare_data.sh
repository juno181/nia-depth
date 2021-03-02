#!/bin/bash
python make_hdf5.py --dataset I256 --batch_size 256 --data_root /mnt/dataset/kitti_depth_train
python calculate_inception_moments.py --dataset I256 --batch_size 256 --data_root /mnt/dataset/kitti_depth_train
