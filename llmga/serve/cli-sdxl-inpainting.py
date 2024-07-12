import argparse
import torch

from llmga.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llmga.llava.conversation import conv_templates, SeparatorStyle
from llmga.llava.model.builder import load_pretrained_model
from llmga.llava.utils import disable_torch_init
from llmga.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from llmga.diffusers.pipeline_stable_diffusion_xl_inpaint_lpw import StableDiffusionXLInpaintPipeline
import os
import copy
import numpy as np


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def read_image(args,image_processor):
    image = load_image(args.image_file)
    mask = Image.open(args.mask_file)

    image_np=np.array(image)
    mask_np = np.array(mask)[:,:,None]
    mask_np[mask_np<125]=0
    mask_np[mask_np>=125]=1

    masked_image_np=image_np*(1-mask_np)
    masked_image = Image.fromarray(masked_image_np.astype('uint8'))

    width=masked_image.width
    height=masked_image.height
    if height>width:
        image_tensor=masked_image.resize((height, height),Image.ANTIALIAS)
    else: 
        image_tensor=masked_image.resize((width, width),Image.ANTIALIAS)
    image_tensor = image_processor.preprocess(image_tensor, return_tensors='pt')['pixel_values'].half().cuda()
    return image_tensor, image, mask


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.sdmodel_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    device = model.device
    pipe = pipe.to(device)

    os.makedirs(args.save_path,exist_ok=True)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif 'llama3' in model_name.lower():
        conv_mode = "llama_3"
    elif "gemma" in model_name.lower():
        conv_mode = "gemma" 
    elif "qwen2" in  model_name.lower():
        conv_mode = "qwen_2"  
    elif "phi-3" in  model_name.lower():
        conv_mode = "phi_3"  
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image_tensor, image, mask = read_image(args,image_processor)
    image_size = image.size
    init_image = image
    cnt=0
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if conv_mode == "gemma":
            stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()

        conv.messages[-1][-1] = outputs
        if conv_mode == "gemma":
            outputs=copy.deepcopy(outputs)[:-19]
        elif conv_mode == "llama_3":
            outputs=copy.deepcopy(outputs)[:-10]
        elif conv_mode == "llava_v1":
            outputs=copy.deepcopy(outputs)[:-4]
        elif conv_mode == "phi_3":
            outputs=copy.deepcopy(outputs)[:-7]
        elif conv_mode == "qwen_2":
            outputs=copy.deepcopy(outputs)[:-10]
        elif conv_mode == "mistral_instruct":
            outputs=copy.deepcopy(outputs)[:-5]
        else:
            outputs=copy.deepcopy(outputs)

        caption=copy.deepcopy(outputs)
        
        id1=caption.find("<gen_image>")
        num_space=12
        id2=caption.find("</gen_image>")
        if id1==-1 and id2==-1:
            caption = caption
        elif id1==-1 and id2!=-1:
            caption = caption[:id2]
        elif id1!=-1 and id2==-1:
            caption = caption[id1+num_space:]
        else:
            caption = caption[id1+num_space:id2]
        if id1==-1:
            print(outputs)
        else:
            print(caption)
            
        if id1!=-1:
            img = pipe(prompt=caption, image=init_image, mask_image=mask,num_inference_steps=30, strength=0.80).images[0]
            img.save(os.path.join(args.save_path,"%03d.png"%(cnt)))
            cnt+=1
        
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--sdmodel_id", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--mask-file", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
