
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118  https_proxy=http://sys-proxy-rd-relay.byted.org:8118  no_proxy=code.byted.org

CUDA_VISIBLE_DEVICES=0 python3 -m llmga.serve.cli \
    --model-path /mnt/bn/wyt-large-dataset/xiabin-model/llmga/checkpoints/llmga-mistral_instruct-full-finetune \
    --image-file ./llmga/serve/examples/jiateng.png \
    # --load-4bit
