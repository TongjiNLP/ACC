cd src_cls
data_dir="../data/cls_data"
output_dir="./output_cls"
CUDA_VISIBLE_DEVICES=0 python run_cls.py \
    --model_name_or_path ../MODEL/roberta \
    --model_type roberta_duma \
    --task_name SpanClassify \
    --do_train \
    --do_eval \
    --overwrite_cache \
    --data_dir $data_dir \
    --output_dir $output_dir \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 8 \
    --max_seq_length 512 \
    --logging_steps 1000 \
    --save_step 1000 \
    --seed 444