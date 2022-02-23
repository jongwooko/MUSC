#!/bin/bash
CUDA_VISIBLE_DEVICES=0
######################################################
# --adapt_trn_languages: chinese, french, german, japanese, korean, spanish
# --adapt_num_shots: 1, 2, 4, 8
# --group_index: 0~39
######################################################

python adapt_training.py --dataset_name pawsx \
                         --adapt_trn_languages german \
                         --adapt_epochs 50 \
                         --adapt_batch_size 32 \
                         --adapt_num_shots 1 \
                         --group_index 0 \
                         --adapt_lr 1e-5 \
                         --train_all_params f \
                         --train_classifier t \
                         --train_pooler f \
                         --reinit_classifier f \
                         --load_ckpt t \
                         --ckpt_path "./checkpoint_baseline/pawsx/debug/1645085231_model_task-pawsx_flr-1.0E-05_ftbs-32_ftepcs-10_sd-3_trnfast-False_evalevery-50_tlang-en_vlang-en/state_dicts/best_state.pt" \
                         --early_stop f \
                         --override False \
                         --train_fast False \
                         --world 0