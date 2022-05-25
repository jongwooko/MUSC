#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python finetuning_baseline.py --dataset_name mldoc \
                              --experiment ce_origin_trans \
                              --trn_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --eval_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.0
