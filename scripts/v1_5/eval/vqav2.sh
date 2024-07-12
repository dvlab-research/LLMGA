#!/bin/bash

sudo chmod -R 777 ./playground/data/eval
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llmga-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llmga.llava.eval.model_vqa_loader  \
#         --model-path /mnt/bn/xiabinpaintv2/CVPR2024/res/LLMGA1.5-v94/llmga-vicuna-7b-v1.5-full-finetune \
#         --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#         --image-folder /mnt/bn/xiabinpaintv2/CVPR2024/rebuttal/eval_dataset/vqav2/test2015 \
#         --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

