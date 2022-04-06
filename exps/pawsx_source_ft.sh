#!/bin/bash
# CUDA_VISIBLE_DEVICES=0

# python -m torch.distributed.launch --nproc_per_node=4 \
#         finetuning_baseline.py --dataset_name pawsx \
#         --trn_languages english \
#         --eval_languages english,german,chinese,french,japanese,korean,spanish \
#         --finetune_epochs 10 \
#         --finetune_batch_size 32 \
#         --eval_every_batch 10 \
#         --override False \
#         --train_fast False \
#         --world 0 \
#         --finetune_lr 1e-5 \
#         --model bert-base-multilingual-cased \

CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py \
                              --dataset_name pawsx \
                              --experiment mislabel_ratio_0.0 \
                              --trn_languages english \
                              --eval_languages english \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 500 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --finetune_lr 1e-5 \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.0
                              
CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py \
                              --dataset_name pawsx \
                              --experiment mislabel_ratio_0.2 \
                              --trn_languages english \
                              --eval_languages english \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 500 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --finetune_lr 1e-5 \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.2

CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py \
                              --dataset_name pawsx \
                              --experiment mislabel_ratio_0.3 \
                              --trn_languages english \
                              --eval_languages english \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 500 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --finetune_lr 1e-5 \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.3
                              
CUDA_VISIBLE_DEVICES=0 python finetuning_baseline.py \
                              --dataset_name pawsx \
                              --experiment mislabel_ratio_0.4 \
                              --trn_languages english \
                              --eval_languages english \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 500 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --finetune_lr 1e-5 \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.4