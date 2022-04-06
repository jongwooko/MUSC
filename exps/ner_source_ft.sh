#!/bin/bash

# 1 epoch = 625 batches (batch size: 32)

CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name panx \
                     --experiment source_epoch \
                     --trn_languages english \
                     --eval_languages english,afrikaans,arabic,bulgarian,bengali,german,greek,spanish,estonian,basque,persian,finnish,french,hebrew,hindi,hungarian,indonesian,italian,japanese,javanese,georgian,kazakh,korean,malayalam,marathi,malay,burmese,dutch,portuguese,russian,swahili,tamil,telugu,thai,tagalog,turkish,urdu,vietnamese,yoruba,chinese \
                     --finetune_epochs 10 \
                     --finetune_batch_size 32 \
                     --eval_every_batch 625 \
                     --override False \
                     --train_fast True \
                     --max_seq_len 128 \
                     --world 0 \
                     --finetune_lr 1e-5 \
                     --model bert-base-multilingual-cased
                     
# python finetuning_baseline.py --dataset_name panx \
#                               --trn_languages english \
#                               --eval_languages english \
#                               --finetune_epochs 10 \
#                               --finetune_batch_size 32 \
#                               --eval_every_batch 50 \
#                               --override False \
#                               --train_fast False \
#                               --world 0 \
#                               --finetune_lr 1e-5
                              