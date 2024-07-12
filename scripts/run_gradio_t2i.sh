
python3 llmga/serve/gradio_t2i_server.py \
    --model_path /mnt/bn/wyt-large-dataset/xiabin-model/llmga/checkpoints/llmga-vicuna-7b-v1.5-full-finetune  \
    --sdmodel_id /mnt/bn/wyt-large-dataset/model/SDXL \
    --lora /mnt/bn/wyt-large-dataset/model/hyper-sd/Hyper-SDXL-1step-Unet.safetensors \
    --model-list-mode reload \
    --port 8334 \
    --load-4bit \