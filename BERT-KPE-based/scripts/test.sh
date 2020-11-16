export DATA_PATH=../../data

CUDA_VISIBLE_DEVICES=2,3 python test.py --run_mode test \
    --model_class bert2span \
    --pretrain_model_type bert-base-cased \
    --dataset_class openkp \
    --per_gpu_test_batch_size 64 \
    --preprocess_folder $DATA_PATH/prepro_dataset \
    --pretrain_model_path $DATA_PATH/pretrain_model \
    --cached_features_dir $DATA_PATH/cached_features \
    --eval_checkpoint /usr0/home/yansenwa/courses/11747/project/BERT-KPE/checkpoints/bert2span/bert2span.openkp.bert.checkpoint \
    --local_rank -1
