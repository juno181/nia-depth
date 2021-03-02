#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--data_root /data/datasets/kitti --experiment_name msg_chn_s2d \
--dataset Kitti_train --model msg_chn --shuffle --toggle_grads --num_workers 8 --model_batch_size 16 --batch_size 16 \
--M_lr 0.001 --M_B2 0.900 \
--M_nl relu \
--adam_eps 1e-8 \
--M_eval_mode \
--M_init He --skip_init \
--num_epochs 50 \
--test_every 1 --save_every 1 --num_best_copies 3 --num_save_copies 2 --seed 0 \
--save_code \
--augment \
