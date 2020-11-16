export TRAIN_DATA_DIR=../data/sample/
export CACHE_DATA_DIR=../data/cache_metaseq/
export OUTPUT_DIR=../output/meta_simple_pred_4_new_mask/
export PRINT_DIR=../output/meta_simple_pred_4_new_mask/
export META_DIR=../metadata/

CUDA_VISIBLE_DEVICES=0 python3 -u run_model.py \
    --cached_features_dir $CACHE_DATA_DIR/ \
    --data_dir $TRAIN_DATA_DIR/ \
    --output_dir $OUTPUT_DIR/ \
    --print_dir $PRINT_DIR/ \
    --meta_dir $META_DIR/ \
    --train \
    --dev \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --batch_size 2 \
    --tag_num 5 \
    --main_metric F@3 \
    --gradient_accumulation_steps 8 \
    --max_text_length 512 \
    --logging_steps 500 \
    --save_steps 2000 \
    --evaluate_during_training \
    --save_best \
    --num_trans 1 \
    --read_from_cached_features \
    #--from_checkpoint $OUTPUT_DIR/checkpoint-best/
