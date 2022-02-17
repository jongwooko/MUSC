#!/bin/bash
CUDA_VISIBLE_DEVICES=0

python finetuning_baseline.py --dataset_name panx \
                              --trn_languages english \
                              --eval_languages english \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast False \
                              --world 0