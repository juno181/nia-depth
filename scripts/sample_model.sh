#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python -W ignore sample.py \
--data_root /ndata/datasets/nia-all --resume_experiment_name Unet_nia_finetune \
--load_weights copy0 \
--dataset nia_train --model Unet --shuffle --toggle_grads --num_workers 8 --model_batch_size 8 --batch_size 8 \
--M_lr 0.001 --M_B2 0.900 \
--M_nl relu \
--adam_eps 1e-8 \
--M_eval_mode \
--M_init He --skip_init \
--num_epochs 50 \
--test_every 1 --save_every 1 --num_best_copies 3 --num_save_copies 2 --seed 0 \
--save_code \
--augment \
