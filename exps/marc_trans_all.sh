#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python finetuning_baseline.py --dataset_name marc \
                              --experiment ce_origin_trans \
                              --trans_train \
                              --trn_languages german,chinese,french,japanese,spanish \
                              --eval_languages english \
                              --finetune_epochs 2 \
                              --finetune_batch_size 16 \
                              --eval_every_batch 300 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.0