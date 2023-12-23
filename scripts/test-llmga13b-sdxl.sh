python3 -m llmga.serve.cli-sdxl \
    --model-path ./checkpoints/Inference/llmga-llama-2-13b-chat-full-finetune  \
    --sdmodel_id ./checkpoints/Inference/llmga-sdxl-t2i \
    --save_path ./res/t2i/llmga13b-sdxl \
    --image-file /PATHtoIMG
