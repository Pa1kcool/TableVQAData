#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=4
DIR=$(pwd)

MODEL="Qwen/Qwen-VL-Chat" # Adjust as needed
DATA="../ETQA/converteddata3.json" # Your data path
OUTPUT_DIR="${DIR}/output_qwen" # Output directory

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Assuming you want to use both GPUs without specifying CUDA_VISIBLE_DEVICES
# PyTorch will automatically use all available GPUs with DataParallel if not otherwise specified

python finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 True \
    --fix_vit True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True


