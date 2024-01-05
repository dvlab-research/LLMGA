#!/bin/bash

MODEL_VERSION="llama-2-13b-chat"

deepspeed --master_port=7000 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/Training/llava-$MODEL_VERSION-full-finetune \
    --version llava_llama_2 \
    --data_path ./data/LLMGA-dataset/llava_instruct_150k.json \
    --data_path2 ./data/LLMGA-dataset/coco2017_train.json \
    --data_path3 ./data/LLMGA-dataset/LAION \
    --image_folder ./data/COCO/train2017 \
    --image_folder2 ./data/LAION-Aesthetic \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_output_start_end True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./work_dirs/llmga-$MODEL_VERSION-full-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
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
