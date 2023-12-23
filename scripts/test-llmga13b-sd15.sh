python3 -m llmga.serve.cli-sd15 \
    --model-path ./checkpoints/Inference/Inference/llmga-llama-2-13b-chat-full-finetune  \
    --sdmodel_id ./checkpoints/Inference/Inference/llmga-sd15-t2i \
    --save_path ./res/t2i/llmga13b-sd15 \
    --image-file /PATHtoIMG
