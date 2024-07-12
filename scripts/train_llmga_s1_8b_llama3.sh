
MODEL_VERSION="llama3-8b-it"

deepspeed --master_port=7001 llmga/llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./base_models/Meta-Llama-3-8B-Instruct \
    --version llama_3 \
    --data_path ./data/jsons/llava_v1_5_mix665k.json \
    --data_path2 ./data/jsons/llmga-data \
    --data_path3 ./data/jsons/text-data \
    --image_folder ./data/llava-imgs \
    --image_folder2 ./data/llmga-imgs \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llmga-llama3-8b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_output_start_end False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mnt/bn/xiabinpaintv2/CVPR2024/res/LLMGA1.5-v101/llmga-$MODEL_VERSION-full-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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