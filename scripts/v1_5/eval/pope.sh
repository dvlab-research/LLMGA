#!/bin/bash

sudo chmod -R 777 ./playground/data/eval
python3 -m llmga.llava.eval.model_vqa_loader \
    --model-path /mnt/bn/xiabinpaintv2/CVPR2024/res/LLMGA1.5-v94/llmga-vicuna-7b-v1.5-full-finetune \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /mnt/bn/xiabinpaintv2/CVPR2024/rebuttal/eval_dataset/coco2014/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llmga-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python3 llmga/llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llmga-7b.jsonl
