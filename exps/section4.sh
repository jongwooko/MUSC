#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name marc \
                              --experiment mt_baseline \
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
                              --model bert-base-multilingual-cased

CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name mldoc \
                              --experiment mt_baseline_1000 \
                              --train_mt \
                              --trn_languages japanese,chinese,french,german,spanish,russian,italian \
                              --eval_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased

CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name mldoc \
                              --experiment mt_baseline_10000 \
                              --train_mt \
                              --trn_languages japanese,chinese,french,german,spanish,russian,italian \
                              --eval_languages english,japanese,chinese,french,german,spanish,russian,italian \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast False \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased
                              
                              
CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name cls \
                              --domain books \
                              --experiment books_mt_baseline \
                              --train_bt \
                              --trn_languages german,french,japanese \
                              --eval_languages english,german,french,japanese \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased
                              
CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name cls \
                              --domain dvd \
                              --experiment dvd_mt_baseline \
                              --train_mt \
                              --trn_languages german,french,japanese \
                              --eval_languages english,german,french,japanese \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased
                              
CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name cls \
                              --domain music \
                              --experiment music_mt_baseline \
                              --train_mt \
                              --trn_languages german,french,japanese \
                              --eval_languages english,german,french,japanese \
                              --finetune_epochs 10 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 50 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased
                              
CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name pawsx \
                              --experiment mt_baseline \
                              --use_cache \
                              --trans_train \
                              --trn_languages chinese,french,german,japanese,korean,spanish \
                              --eval_languages english,chinese,french,german,japanese,korean,spanish \
                              --finetune_epochs 4 \
                              --finetune_batch_size 32 \
                              --eval_every_batch 300 \
                              --override False \
                              --train_fast False \
                              --max_seq_len 128 \
                              --world 0 \
                              --finetune_lr 1e-5 \
                              --model bert-base-multilingual-cased
                              
CUDA_VISIBLE_DEVICES=5 python finetuning_baseline.py --dataset_name xnli \
                             --experiment mt_baseline \
                             --use_cache \
                             --trans_train \
                             --trn_languages arabic,bulgarian,chinese,french,german,greek,hindi,russian,spanish,swahili,thai,turkish,urdu,vietnamese \
                             --eval_languages english,arabic,bulgarian,chinese,french,german,greek,hindi,russian,spanish,swahili,thai,turkish,urdu,vietnamese \
                             --finetune_epochs 2 \
                             --finetune_batch_size 32 \
                             --eval_every_batch 3000 \
                             --override False \
                             --train_fast False \
                             --max_seq_len 128 \
                             --world 0 \
                             --finetune_lr 3e-5 \
                             --model bert-base-multilingual-cased