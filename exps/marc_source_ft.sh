#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name marc \
                              --experiment debug \
                              --trn_languages english \
                              --eval_languages english \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 300 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.0