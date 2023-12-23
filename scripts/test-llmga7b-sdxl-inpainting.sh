python3 -m llmga.serve.cli-sdxl-inpainting \
    --model-path ./checkpoints/Inference/llmga-llama-2-7b-chat-full-finetune  \
    --sdmodel_id ./checkpoints/Inference/llmga-sdxl-inpainting \
    --save_path ./res/inpainting/llmga7b-sdxl \
    --image-file /PATHtoIMG \
    --mask-file /PATHtomask


