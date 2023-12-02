python3 -m llmga.serve.cli-sdxl-inpainting \
    --model-path ./checkpoints/llmga-llama-2-7b-chat-full-finetune  \
    --sdmodel_id ./checkpoints/llmga-sdxl-inpainting \
    --save_path ./exp-inpainting/llmga7b-sdxl \
    --image-file /PATHtoIMG \
    --mask-file /PATHtomask


