import argparse
import torch

from llmga.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llmga.llava.conversation import conv_templates, SeparatorStyle
from llmga.llava.model.builder import load_pretrained_model
from llmga.llava.utils import disable_torch_init
from llmga.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


import requests
from PIL import Image
from io import BytesIO
import copy

from llmga.diffusers.pipeline_semantic_stable_diffusion_img2img_solver_lpw_mask import SemanticStableDiffusionImg2ImgPipeline_DPMSolver
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from llmga.diffusers.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
import random
import cv2
import PIL
from PIL import Image
import os
import numpy as np

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def randomize_seed_fn(seed, is_random):
    if is_random:
        seed = random.randint(0, np.iinfo(np.int32).max)
    return seed

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def crop_image(image):
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    pipe = SemanticStableDiffusionImg2ImgPipeline_DPMSolver.from_pretrained(args.sd_model_id,vae=vae,torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(args.sd_model_id, subfolder="scheduler"
                                                             , algorithm_type="sde-dpmsolver++", solver_order=2)

    def sample(zs, wts, mask_image, attention_store, text_cross_attention_maps, prompt_tar="", cfg_scale_tar=15, skip=36, eta=1):
        latents = wts[-1].expand(1, -1, -1, -1)
        img, attention_store, text_cross_attention_maps = pipe(
            prompt=prompt_tar,
            init_latents=latents,
            guidance_scale=cfg_scale_tar,
            mask_image=mask_image,
            attention_store = attention_store, text_cross_attention_maps=text_cross_attention_maps,
            zs=zs,
        )
        return img.images[0], attention_store, text_cross_attention_maps


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

    image = load_image(args.image_file)

    image=np.array(image)
    image_np=crop_image(image)
    image=Image.fromarray(image_np)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    image_size = image.size
    for i in range(2):
        if i==0: 
            inp="Generate a similar image"
        else:
            inp = input(f"{roles[0]}: ")

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
            outputs=caption
            print(caption)
        else:
            print(outputs)
        
        if i==0: 
            src_prompt = caption
        else: 
            tar_prompt = caption
    
    init_image = Image.fromarray(image_np)
    mask_image = pipe.generate_mask(image=init_image, source_prompt=src_prompt, target_prompt=tar_prompt,mask_thresholding_ratio=3.0)

    cv2.imwrite(os.path.join(args.save_path,"mask.png"),mask_image[0]*255)    
    zs_tensor, wts_tensor = pipe.invert(
        image_path = image_np,
        source_prompt =src_prompt,
        source_guidance_scale= args.src_cfg_scale,
        num_inversion_steps = args.steps,
        skip = args.skip,
        eta = 1.0,
    )
    wts = wts_tensor
    zs = zs_tensor
    pure_ddpm_img, attention_store, text_cross_attention_maps = sample(zs, wts, mask_image, attention_store=None, text_cross_attention_maps=None, prompt_tar=tar_prompt, skip=args.skip, cfg_scale_tar=args.tar_cfg_scale)
    Image.fromarray(image_np).save(os.path.join(args.save_path,"input_image.png"))
    pure_ddpm_img.save(os.path.join(args.save_path,"edited_image.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    #**************SD Augs*****************
    parser.add_argument("--sd_model_id", type=str, required=True)
    parser.add_argument("--src_cfg_scale", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--skip", type=int, default=25)
    parser.add_argument("--tar_cfg_scale", type=float, default=7.5)
    args = parser.parse_args()
    main(args)
