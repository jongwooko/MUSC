#!/bin/bash                              
CUDA_VISIBLE_DEVICES=1 python finetuning_baseline.py --dataset_name cls \
                              --experiment cls_debug \
                              --trn_languages english \
                              --eval_languages english,german,french,japanese \
                              --finetune_epochs 10 \
                              --finetune_batch_size 8 \
                              --eval_every_batch 100 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --max_seq_len 256 \
                              --model bert-base-multilingual-cased \
                              --mislabel_type uniform \
                              --mislabel_ratio 0.0