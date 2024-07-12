import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from llmga.llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llmga.llava.constants import LOGDIR
from llmga.llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
from llmga.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llmga.llava.conversation import conv_templates, SeparatorStyle
from llmga.llava.model.builder import load_pretrained_model
from llmga.llava.utils import disable_torch_init
from llmga.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import hashlib
import torch
from llmga.diffusers.pipeline_stable_diffusion_xl_lpw import StableDiffusionXLPipeline
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
import copy
from safetensors.torch import load_file

headers = {"User-Agent": "LLMGA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)




get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""



def load_demo(url_params, request: gr.Request):

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(
                value=model, visible=True)

    state = default_conversation.copy()
    gen_image_state=[[],0]
    return state, gen_image_state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    gen_image_state=[[],0]
    return state,gen_image_state, dropdown_update



def regenerate(state, gen_image_state, image_process_mode, request: gr.Request):
    # logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    
    return (state, gen_image_state, state.to_gradio_chatbot(gen_image_state[0]), "", None) + (disable_btn,) * 3


def regenerate_img(state, gen_image_state, image_process_mode, request: gr.Request):
    # logger.info(f"regenerate. ip: {request.client.host}")
    gen_image_state[0]=gen_image_state[0][:gen_image_state[1]]
    
    return (state, gen_image_state, state.to_gradio_chatbot(gen_image_state[0]), "", None) + (disable_btn,) * 3


def clear_history(request: gr.Request):
    # logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    gen_image_state=[[],0]
    return (state, gen_image_state, state.to_gradio_chatbot(gen_image_state[0]), "", None, None) + (disable_btn,) * 3


def add_text(state, gen_image_state, text, image, image_process_mode, request: gr.Request):
    # logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(gen_image_state[0]), "", None) + (no_change_btn,) * 3
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(gen_image_state[0]), moderation_msg, None) + (
                no_change_btn,) * 3

    text = text[:1536]  # Hard cut-off
    
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            if model.config.mm_use_im_start_end:
                text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
            else:
                text = DEFAULT_IMAGE_TOKEN + '\n' + text
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    tp=state.to_gradio_chatbot(gen_image_state[0])
    
    for tpp in tp:
        if tpp[-1] is None:
            continue
        tpp[-1] = tpp[-1].replace("\n\n","\n")
        tpp[-1] = tpp[-1][4:-4]
        tpp[-1]=tpp[-1].replace("<gen_image>", "\n\n( The generation prompt: ")
        tpp[-1]=tpp[-1].replace("</gen_image>", ")\n\n")
        # if "<gen_image>" in tpp[-1] and "</gen_image>" in tpp[-1]:
        #     tpp[-1]="The generation is finished: \n\n" + tpp[-1]
    return (state, gen_image_state, tp, "", None) + (disable_btn,) * 3


def http_bot(state, gen_image_state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    # logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, gen_image_state, state.to_gradio_chatbot(gen_image_state[0])) + (no_change_btn,) * 3
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llmga" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            elif 'llama3' in model_name.lower():
                template_name = "llama_3"
            elif "gemma" in model_name.lower():
                template_name = "gemma" 
            elif "qwen2" in  model_name.lower():
                template_name = "qwen_2"  
            elif "phi-3" in  model_name.lower():
                template_name = "phi_3"  
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
            
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    prompt = state.get_prompt()

    images = state.get_images(return_pil=True)
    image_tensors =[image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda() for image in images]
    if len(image_tensors)==0:
        image_tensor=None
    elif len(image_tensors)==1:
        image_tensor=image_tensors[0]
    else:
        image_tensor=image_tensors
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = state.sep if state.sep_style != SeparatorStyle.TWO else state.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=float(temperature),
            max_new_tokens=1024,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id)
    
    outputs = tokenizer.decode(output_ids[0]).strip()

    state.messages[-1][-1] = outputs
    
        # if "<gen_image>" in tpp[-1] and "</gen_image>" in tpp[-1]:
        #     tpp[-1]="The generation is finished: \n\n" + tpp[-1]
    tp=state.to_gradio_chatbot(gen_image_state[0])
    for tpp in tp:
        if tpp[-1] is None:
            continue
        tpp[-1] = tpp[-1].replace("\n\n","\n")
        if state.version == "gemma":
            tpp[-1] = tpp[-1][:-19]
        elif state.version == "llama_v3":
            tpp[-1] = tpp[-1][:-10]
        elif state.version == "v1":
            tpp[-1] = tpp[-1][4:-4]
        elif state.version == "phi_3":
            tpp[-1] = tpp[-1][:-7]
        elif state.version == "qwen_2":
            tpp[-1] = tpp[-1][:-10]
        elif state.version == "llama_v2":
            tpp[-1] = tpp[-1][:-5]
        else:
            pass
        # tpp[-1] = tpp[-1][4:-4]
        tpp[-1]=tpp[-1].replace("<gen_image>", "\n\n( The generation prompt: ")
        tpp[-1]=tpp[-1].replace("</gen_image>", ")\n\n") 
    
    yield (state,gen_image_state,tp) + (enable_btn,) * 3
    

def generation_bot(state,gen_image_state,num_inference_steps):
    
    global image
    gen_image_state[1]=len(gen_image_state[0])
    outputs = state.messages[-1][-1]
    outputs=copy.deepcopy(outputs)
    caption=copy.deepcopy(outputs)
    fg=False
    while(True):
        id1=outputs.find("<gen_image>")
        if id1==-1:
            break
        fg=True
        num_space=12
        id2=outputs.find("</gen_image>")
        if id2==-1:
            caption = outputs[id1+num_space:]
            outputs = ""
        else:
            caption = outputs[id1+num_space:id2]
            outputs = outputs[id2+12:]

        image=pipe(prompt=caption, num_inference_steps=1, guidance_scale=0, timesteps=[800]).images[0]
        gen_image_state[0].append(image)
        
        # image = pipe(caption,num_inference_steps=num_inference_steps).images[0]
    tp=state.to_gradio_chatbot(gen_image_state[0])
    for tpp in tp:
        if tpp[-1] is None:
            continue
        tpp[-1] = tpp[-1].replace("\n\n","\n")
        if state.version == "gemma":
            tpp[-1] = tpp[-1][:-19]
        elif state.version == "llama_v3":
            tpp[-1] = tpp[-1][:-10]
        elif state.version == "v1":
            tpp[-1] = tpp[-1][4:-4]
        elif state.version == "phi_3":
            tpp[-1] = tpp[-1][:-7]
        elif state.version == "qwen_2":
            tpp[-1] = tpp[-1][:-10]
        elif state.version == "llama_v2":
            tpp[-1] = tpp[-1][:-5]
        else:
            pass
        tpp[-1]=tpp[-1].replace("<gen_image>", "\n\n( The generation prompt: ")
        tpp[-1]=tpp[-1].replace("</gen_image>", ")\n\n") 
    if fg:    
        yield (state,gen_image_state, tp, image, enable_btn)
    else:
        yield (state,gen_image_state, tp, None, enable_btn)


title_markdown = ("""
# LLMGA: Multimodal Large Language Model based Generation Assistant
[[Project Page](https://llmga.github.io/)] [[Code](https://github.com/dvlab-research/LLMGA)] [[Model](https://huggingface.co/binxia)] | 📚 [[LLMGA](https://arxiv.org/pdf/2311.16500.pdf)] 
""")



block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode,concurrency_count=10):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLMGA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        gen_image_state = gr.State(value=[[],0])
        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Resize",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/jiateng.png", "Generate a similar image."],
                    [f"{cur_dir}/examples/snow_scene.jpeg", "Generate a similar image."],
                    [f"{cur_dir}/examples/aes.png", "Generate a similar image."],
                ], inputs=[imagebox, textbox])
                gr.Examples(examples=[
                    ["Please help me design a spaceship and show me its blueprint."],
                    ["I need a Chinese style painting for decoration. Please help me design it."]
                ], inputs=[textbox])
                
                

                with gr.Accordion("MLLM's Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=1024, step=64, interactive=True, label="Max output tokens",)
                
                with gr.Accordion("SD's Parameters", open=False) as parameter_row:
                    num_inference_steps = gr.Slider(minimum=10, maximum=80, value=30, step=2, interactive=True, label="num_inference_steps",)
                
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLMGA Chatbot", height=750)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    regenerate_btn = gr.Button(value="🔄 Text Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
            
            with gr.Column(scale=4):
                gen_imagebox = gr.Image(type="pil",label="Image Output", height=400)
                with gr.Row(elem_id="buttons") as button_row:
                    gen_regenerate_btn = gr.Button(value="🔄 Image Regenerate", interactive=False)

    
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [regenerate_btn, clear_btn, gen_regenerate_btn]

        regenerate_btn.click(
            regenerate,
            [state, gen_image_state, image_process_mode],
            [state, gen_image_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, gen_image_state, model_selector, temperature, top_p, max_output_tokens],
            [state, gen_image_state, chatbot] + btn_list,
            # concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, gen_image_state, chatbot, textbox, imagebox, gen_imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, gen_image_state, textbox, imagebox, image_process_mode],
            [state, gen_image_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, gen_image_state, model_selector, temperature, top_p, max_output_tokens],
            [state, gen_image_state, chatbot] + btn_list,
            # concurrency_limit=concurrency_count
        ).then(
            generation_bot,
            [state, gen_image_state, num_inference_steps],
            [state, gen_image_state, chatbot,gen_imagebox,gen_regenerate_btn],
            # concurrency_limit=concurrency_count
        )

        submit_btn.click(
            add_text,
            [state, gen_image_state, textbox, imagebox, image_process_mode],
            [state, gen_image_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, gen_image_state, model_selector, temperature, top_p, max_output_tokens],
            [state, gen_image_state, chatbot] + btn_list,
            queue=False
            # concurrency_limit=concurrency_count
        ).then(
            generation_bot,
            [state, gen_image_state, num_inference_steps],
            [state, gen_image_state, chatbot,gen_imagebox,gen_regenerate_btn],
            queue=False
            # concurrency_limit=concurrency_count
        )

        gen_regenerate_btn.click(
            regenerate_img,
            [state, gen_image_state, image_process_mode],
            [state, gen_image_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            generation_bot,
            [state, gen_image_state, num_inference_steps],
            [state, gen_image_state, chatbot,gen_imagebox,gen_regenerate_btn],
            # concurrency_limit=concurrency_count
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state,gen_image_state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state,gen_image_state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="PathToMLLM")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--sdmodel_id", type=str, default="PathToSD")
    parser.add_argument("--lora", type=str, default="PathTolora")
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency-count", type=int, default=1)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    args = parser.parse_args()
    models= [args.model_path.split("/")[-1]]
    

    disable_torch_init()
    global image
    image=None

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     args.sdmodel_id,variant="fp16", use_safetensors=True,add_watermarker=False
    # )
    unet = UNet2DConditionModel.from_config(args.sdmodel_id, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(args.lora, device="cuda"))
    pipe = DiffusionPipeline.from_pretrained(args.sdmodel_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    device = model.device
    pipe = pipe.to(device)

    demo = build_demo(args.embed,concurrency_count=args.concurrency_count,)
    demo.queue(
        api_open=False
    ).launch(
        server_name="[::]",
        # server_port=args.port,
        share=True,
    )

