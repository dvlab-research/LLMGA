#!/bin/bash
sudo chmod -R 777 ./playground/data/eval
SPLIT="mmbench_dev_20230712"

# python3 -m llmga.llava.eval.model_vqa_mmbench \
#     --model-path /mnt/bn/xiabinpaintv2/CVPR2024/res/LLMGA1.5-v94/llmga-vicuna-7b-v1.5-full-finetune \
#     --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llmga-7b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llmga-7b
