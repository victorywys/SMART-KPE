#!/bin/sh
#SBATCH -N 1 # tasks requested
#SBATCH -n 6 # tasks requested
#SBATCH --gres=gpu:1 #GPU
#SBATCH -e ./tmp/err_doc-title_novis_4 # send stderr to errfile 
#SBATCH -o ./tmp/out_doc-title_novis_4 # send stdout to outfile 
#SBATCH --mem=48000 # memory in Mb
#SBATCH -t 1-00:00 # time:q
#SBATCH -p gpu

export TRAIN_DATA_DIR=../../data_title/
export CACHE_DATA_DIR=../../data_title/title_snap/
export OUTPUT_DIR=../output_new/title_snap_4/
export PRINT_DIR=../output_new/title_snap_4/
export META_DIR=../../metadata/

CUDA_VISIBLE_DEVICES=2,3,4,5 python3 -u run_model.py \
    --cached_features_dir $CACHE_DATA_DIR/ \
    --data_dir $TRAIN_DATA_DIR/ \
    --output_dir $OUTPUT_DIR/ \
    --print_dir $PRINT_DIR \
    --meta_dir $META_DIR \
    --use_snapshot \
    --train \
    --dev \
    --test \
    --num_trans 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --batch_size 32 \
    --tag_num 5 \
    --main_metric F@3 \
    --gradient_accumulation_steps 4 \
    --max_text_length 512 \
    --logging_steps 200 \
    --save_steps 2000 \
    --evaluate_during_training \
    --save_best \
    --include_title \
    --read_from_cached_features \
#    --from_checkpoint $OUTPUT_DIR/checkpoint-best
