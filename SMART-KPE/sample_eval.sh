#!/bin/sh
#SBATCH -N 1 # tasks requested
#SBATCH -n 6 # tasks requested
#SBATCH --gres=gpu:1 #GPU
#SBATCH -e ./tmp/err_eval_openkp1e-5 # send stderr to errfile 
#SBATCH -o ./tmp/out_eval_openkp1e-5 # send stdout to outfile 
#SBATCH --mem=48000 # memory in Mb
#SBATCH -t 1-00:00 # time:q
#SBATCH -p gpu

export TRAIN_DATA_DIR=../../data_title/
export CACHE_DATA_DIR=../../data_title/title_snap
export OUTPUT_DIR=../output/me/
export PRINT_DIR=../output/meta_simple_pred_4_new_mask/
export META_DIR=../../openkp_meta/

python3 -u run_model.py \
    --cached_features_dir $CACHE_DATA_DIR/ \
    --data_dir $TRAIN_DATA_DIR/ \
    --output_dir $OUTPUT_DIR/ \
    --meta_dir $META_DIR/ \
    --use_snapshot \
    --dev \
    --batch_size 4 \
    --tag_num 5 \
    --max_text_length 512 \
    --from_checkpoint $OUTPUT_DIR/checkpoint-best/ \
    --print_dir $PRINT_DIR \
    --read_from_cached_features \
