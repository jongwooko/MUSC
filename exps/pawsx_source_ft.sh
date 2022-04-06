#!/bin/bash

# 1 epoch = 1544 batches (batch size: 32)

CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name pawsx \
                                                     --experiment source_epoch \
                                                     --trn_languages english \
                                                     --eval_languages english,german,chinese,french,japanese,korean,spanish \
                                                     --finetune_epochs 10 \
                                                     --finetune_batch_size 32 \
                                                     --eval_every_batch 1544 \
                                                     --override False \
                                                     --train_fast True \
                                                     --max_seq_len 128 \
                                                     --world 0 \
                                                     --finetune_lr 1e-5 \
                                                     --model bert-base-multilingual-cased \
                                                     --mislabel_type uniform \
                                                     --mislabel_ratio 0.0