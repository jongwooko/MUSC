#!/bin/bash

# 1 epoch = 681 batches (batch size: 32)

CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name udpos \
                     --experiment source_epoch \
                     --trn_languages english \
                     --eval_languages english,afrikaans,arabic,bulgarian,german,greek,spanish,estonian,basque,persian,finnish,french,hebrew,hindi,hungarian,indonesian,italian,japanese,korean,marathi,dutch,portuguese,russian,tamil,telugu,turkish,urdu,vietnamese,chinese \
                     --finetune_epochs 10 \
                     --finetune_batch_size 32 \
                     --eval_every_batch 681 \
                     --override False \
                     --train_fast True \
                     --max_seq_len 128 \
                     --world 0 \
                     --finetune_lr 1e-5 \
                     --model bert-base-multilingual-cased
                     
# python finetuning_baseline.py --dataset_name udpos \
#                               --trn_languages english \
#                               --eval_languages english \
#                               --finetune_epochs 10 \
#                               --finetune_batch_size 32 \
#                               --eval_every_batch 50 \
#                               --override False \
#                               --train_fast False \
#                               --world 0 \
#                               --finetune_lr 1e-5
                              