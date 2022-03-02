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

python finetuning_baseline.py --dataset_name pawsx \
                              --trn_languages english \
                              --eval_languages english,german,chinese,french,japanese,korean,spanish \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast True \
                              --max_seq_len 128 \
                              --world 0 \
                              --finetune_lr 1e-5