cd src_cor
data_dir="../data/cor_data"
output_dir="./output_cor"
CUDA_VISIBLE_DEVICES=0 python run.py \
    --do_train \
    --do_eval \
    --do_lower_case \
    --overwrite_output \
    --overwrite_cache \
    --eval_all_checkpoints \
    --task_name squad \
    --per_gpu_eval_batch_size 16 \
    --max_seq_length 512 \
    --model_type roberta \
    --model_name_or_path ../MODEL/roberta \
    --data_dir $data_dir \
    --learning_rate 3e-5 \
    --num_train_epochs 12 \
    --output_dir $output_dir \
    --per_gpu_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 100 \
    --save_steps 1000 \
    --logging_steps 500 \
    --seed 333