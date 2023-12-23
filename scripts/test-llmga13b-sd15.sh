python3 -m llmga.serve.cli-sd15 \
    --model-path ./checkpoints/Inference/llmga-llama-2-13b-chat-full-finetune  \
    --sdmodel_id ./checkpoints/Inference/llmga-sd15-t2i \
    --save_path ./exp/llmga13b-sd15 \
    --image-file /PATHtoIMG
