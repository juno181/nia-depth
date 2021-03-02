#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--data_root /data/datasets/kitti --experiment_name cspn_s2d \
--dataset nipa_train --model cspn --shuffle --toggle_grads --num_workers 8 --model_batch_size 8 --batch_size 8 \
--M_lr 0.0001 --M_B2 0.900 \
--M_nl relu \
--adam_eps 1e-8 \
--M_eval_mode \
--M_init He --skip_init \
--num_epochs 50 \
--test_every 1 --save_every 1 --num_best_copies 3 --num_save_copies 2 --seed 0 \
--save_code \
--augment \
