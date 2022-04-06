#!/bin/bash

# 1 epoch = 12272 batches (batch size: 32)

CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name xnli \
                     --experiment source_epoch \
                     --trn_languages english \
                     --eval_languages english,arabic,bulgarian,chinese,french,german,greek,hindi,russian,spanish,swahili,thai,turkish,urdu,vietnamese \
                     --finetune_epochs 10 \
                     --finetune_batch_size 32 \
                     --eval_every_batch 12272 \
                     --override False \
                     --train_fast True \
                     --max_seq_len 128 \
                     --world 0 \
                     --finetune_lr 3e-5 \
                     --model bert-base-multilingual-cased

# python finetuning_baseline.py --dataset_name xnli \
#                               --trn_languages english \
#                               --eval_languages english \
#                               --finetune_epochs 10 \
#                               --finetune_batch_size 32 \
#                               --eval_every_batch 50 \
#                               --override False \
#                               --train_fast False \
#                               --world 0 \
#                               --finetune_lr 3e-5
                              