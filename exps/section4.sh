#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name marc \
                              --experiment or_baseline \
                              --trn_languages english,german,chinese,french,japanese,spanish \
                              --eval_languages english,german,chinese,french,japanese,spanish \
                              --finetune_epochs 4 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 300 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased
                              
CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name marc \
                              --experiment bt_baseline \
                              --train_bt \
                              --trn_languages english,german,chinese,french,japanese,spanish \
                              --eval_languages english,german,chinese,french,japanese,spanish \
                              --finetune_epochs 4 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 300 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased

CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name mldoc \
                              --experiment or_baseline_10000 \
                              --trn_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --eval_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 300 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased
                              
CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py --dataset_name mldoc \
                              --experiment bt_baseline_10000 \
                              --train_bt \
                              --trn_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --eval_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 300 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased