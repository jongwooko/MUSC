#!/bin/bash

# 1 epoch = 12272 batches (batch size: 32)

CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name marc \
                     --experiment marc_supcon \
                     --train_mt \
                     --trn_languages german,chinese,french,japanese,spanish \
                     --eval_languages english,german,chinese,french,japanese,spanish \
                     --finetune_epochs 4 \
                     --finetune_batch_size 32 \
                     --eval_every_batch 300 \
                     --override False \
                     --train_fast False \
                     --world 0 \
                     --finetune_lr 1e-5 \
                     --model bert-base-multilingual-cased \
                     --use_supcon \
                     --lam 0.9
                     
CUDA_VISIBLE_DEVICES=6 python finetuning_baseline.py --dataset_name marc \
                     --experiment marc_mixup \
                     --train_mt \
                     --trn_languages german,chinese,french,japanese,spanish \
                     --eval_languages english,german,chinese,french,japanese,spanish \
                     --finetune_epochs 4 \
                     --finetune_batch_size 32 \
                     --eval_every_batch 300 \
                     --override False \
                     --train_fast False \
                     --world 0 \
                     --finetune_lr 1e-5 \
                     --model bert-base-multilingual-cased \
                     --use_mix
                     
CUDA_VISIBLE_DEVICES=7 python finetuning_baseline.py --dataset_name marc \
                     --experiment marc_supcon_mixup \
                     --train_mt \
                     --trn_languages german,chinese,french,japanese,spanish \
                     --eval_languages english,german,chinese,french,japanese,spanish \
                     --finetune_epochs 4 \
                     --finetune_batch_size 32 \
                     --eval_every_batch 300 \
                     --override False \
                     --train_fast False \
                     --world 0 \
                     --finetune_lr 1e-5 \
                     --model bert-base-multilingual-cased \
                     --use_supcon \
                     --lam 0.9 \
                     --use_mix
                     
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
                              