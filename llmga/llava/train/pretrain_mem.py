from llmga.llava.train.pretrain import train
import os
# os.environ["WANDB_DISABLED"]="true"
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
