#!/bin/bash

# 1 epoch = 1544 batches (batch size: 32)

CUDA_VISIBLE_DEVICES=7 python finetuning_baseline.py --dataset_name pawsx \
                                                     --experiment trans_debug \
                                                     --trans_train \
                                                     --trn_languages german \
                                                     --eval_languages german \
                                                     --finetune_epochs 10 \
                                                     --finetune_batch_size 16 \
                                                     --eval_every_batch 1544 \
                                                     --override False \
                                                     --train_fast False \
                                                     --max_seq_len 128 \
                                                     --world 0 \
                                                     --finetune_lr 1e-5 \
                                                     --model bert-base-multilingual-cased \
                                                     --mislabel_type uniform \
                                                     --mislabel_ratio 0.0