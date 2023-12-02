#!/bin/bash


MODEL_VERSION="llama-2-7b-chat"

deepspeed --master_port=7000 llmga/llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-$MODEL_VERSION-full-finetune \
    --version llava_llama_2 \
    --data_path ./dataset/llava_instruct_150k.json \
    --data_path2 ./dataset/COCO \
    --data_path3 ./dataset/LLM-info \
    --image_folder ./dataset/coco/train2017 \
    --image_folder2 ./dataset/laion_aesthetics \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_output_start_end True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llmga-$MODEL_VERSION-full-finetune \
    --num_train_epochs 6 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --image_aspect_ratio "resizesquare" \
    --report_to wandb
