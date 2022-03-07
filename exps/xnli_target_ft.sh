#!/bin/bash
CUDA_VISIBLE_DEVICES=0
######################################################
# --adapt_trn_languages: 'arabic', 'bulgarian', 'chinese', 'french', 'german', 'greek', 'hindi', 'russian', 'spanish', 'swahili', 'thai', 'turkish', 'urdu', 'vietnamese'
# --adapt_num_shots: 1, 2, 4, 8
# --group_index: 0~39
######################################################

lang_lst="arabic bulgarian chinese french german greek hindi russian spanish swahili thai turkish urdu vietnamese"
group_indices=$(seq 0 39)
shot_lst="1 2 4 8"

for s in $shot_lst
do
    for l in $lang_lst
    do
        for g in $group_indices
        do
            python adapt_training.py --dataset_name xnli \
                                    --adapt_trn_languages $l \
                                    --experiment $l\($s\)-$g \
                                    --adapt_epochs 50 \
                                    --adapt_num_shots $s \
                                    --adapt_batch_size $s \
                                    --group_index $g \
                                    --adapt_lr 1e-5 \
                                    --train_all_params True \
                                    --train_classifier True \
                                    --train_pooler True \
                                    --reinit_classifier False \
                                    --reinit_pooler False \
                                    --load_ckpt True \
                                    --ckpt_path "./checkpoint_baseline/xnli/debug/1646210965_model_task-xnli_flr-3.0E-05_ftbs-32_ftepcs-10_sd-3_trnfast-False_evalevery-2000_tlang-en_vlang-en-ar-bg-zh-fr-de-el-hi-ru-es-sw-th-tr-ur-vi/state_dicts/best_state.pt" \
                                    --early_stop True \
                                    --early_stop_patience 10 \
                                    --override False \
                                    --train_fast True \
                                    --world 0
        done
    done
done