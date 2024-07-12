
sudo chmod -R 777 /mnt/bn/xiabinpaintv2/CVPR2024/code-final/LLMGA-v1/res


CUDA_VISIBLE_DEVICES=1 python3 -m llmga.serve.cli-sd15-editing \
    --model-path /mnt/bn/xiabinpaintv2/CVPR2024/res/LLMGA1.5-v73/llmga-vicuna-7b-v1.5-full-finetune  \
    --image-file /mnt/bn/xiabinpaintv2/CVPR2024/code-final/LLMGA-v1/000000003613_ori.png \
    --save_path /mnt/bn/xiabinpaintv2/CVPR2024/code-final/LLMGA-v1/res \
    --sd_model_id /mnt/bn/inpainting-bytenas-lq/xiabin/new-SD-model/sd15-t2i-outputs-05v10
    # --load-4bit


