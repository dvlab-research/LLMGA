python3 -m llmga.serve.cli-sd15-inpainting \
    --model-path ./checkpoints/llmga-llama-2-7b-chat-full-finetune  \
    --sdmodel_id ./checkpoints/llmga-sd15-inpainting \
    --save_path ./exp-inpainting/llmga7b-sd15 \
    --image-file /PATHtoIMG \
    --mask-file /PATHtomask

