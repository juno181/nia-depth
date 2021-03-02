#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python -W ignore train.py \
--data_root /data/datasets/nia-all --experiment_name cspn_s2d_nia_finetune \
--resume --resume_experiment_name cspn_s2d \
--dataset nia_train --model cspn --shuffle --toggle_grads --num_workers 8 --model_batch_size 16 --batch_size 16 \
--M_lr 0.001 --M_B2 0.900 \
--M_nl relu \
--adam_eps 1e-8 \
--M_eval_mode \
--M_init He --skip_init \
--num_epochs 53 \
--test_every 1 --save_every 1 --num_best_copies 5 --num_save_copies 1 --seed 0 \
--save_code \
