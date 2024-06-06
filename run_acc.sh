cd src_acc
CUDA_VISIBLE_DEVICES=0 python run_acc.py \
    --data_dir ../data/MultiSpanQA_data \
    --output_dir ../output_acc \
    --pred_dir ../predictions/prediction_1.json \
    --task_name MultiSpanQA \
    --answer_tokenizer_path ../MODEL/roberta \
    --model_type roberta \
    --cls_model_path ../MODEL/cls_ckpt \
    --cor_model_path ../MODEL/cor_ckpt \
    --max_length 512 \
    --do_eval \
    --use_cls \
    --use_cor
