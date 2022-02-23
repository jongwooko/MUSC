#!/bin/bash
CUDA_VISIBLE_DEVICES=0
######################################################
# --adapt_trn_languages: chinese, french, german, japanese, korean, spanish
# --adapt_num_shots: 1, 2, 4, 8
# --group_index: 0~39
######################################################

python adapt_training.py --dataset_name panx \
                         --adapt_trn_languages german \
                         --adapt_epochs 50 \
                         --adapt_num_shots 1 \
                         --adapt_batch_size 1 \
                         --group_index 0 \
                         --adapt_lr 1e-5 \
                         --train_all_params False \
                         --train_classifier True \
                         --train_pooler False \
                         --reinit_classifier False \
                         --load_ckpt True \
                         --ckpt_path "./checkpoint_baseline/panx/debug/1645106200_model_task-panx_flr-1.0E-05_ftbs-32_ftepcs-10_sd-3_trnfast-False_evalevery-50_tlang-en_vlang-en/state_dicts/last_state.pt" \
                         --early_stop True \
                         --early_stop_patience 10 \
                         --override False \
                         --train_fast True \
                         --world 0