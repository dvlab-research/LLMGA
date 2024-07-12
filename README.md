


<p align="center" width="10%">
<img src="imgs/logo.png" style="width: 30%" align=center> 
</p>


# LLMGA: Multimodal Large Language Model-based Generation Assistant (ECCV2024)

[Bin Xia](https://scholar.google.com/citations?user=rh2fID8AAAAJ&hl=zh-CN), [Shiyin Wang](), [Yingfan Tao](https://scholar.google.com/citations?user=GYDnPdQAAAAJ&hl=zh-CN&oi=ao), [Yitong Wang](https://scholar.google.com/citations?user=NfFTKfYAAAAJ&hl=zh-CN), and [Jiaya Jia](https://scholar.google.com/citations?user=XPAkzTEAAAAJ&hl=zh-CN&oi=ao)

<a href="https://llmga.github.io/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href="https://arxiv.org/pdf/2311.16500.pdf"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://huggingface.co/binxia'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/binxia/LLMGA-datasetv2/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

## News
**New Version (Accepted by ECCV2024):**
- [x] [2024.07.06] The finetuned SD15 models have been released, including [SD15-T2I](https://huggingface.co/binxia/llmga-sd15-t2i-v2) and [SD15-inpainting](https://huggingface.co/binxia/llmga-sd15-inpainting-v2). Notably, our SD15-T2I model can also be used for instruction-based editing of LLMGA.
- [x] [2024.07.06] The finetuned SDXL models have been released, including [SDXL-T2I](https://huggingface.co/binxia/llmga-sdxl-t2i) and [SDXL-inpainting](https://huggingface.co/binxia/llmga-sdxl-inpainting-v2/tree/main). 
- [x] [2024.07.06] The pre-trained models, which further support Chinese (obtained by further fine-tuned on mixed Chinese and English data),  have been released, including [llmga-cn-vicuna 7b](https://huggingface.co/binxia/llmga-cn-vicuna-7b-v1.5-full-finetune), [llmga-cn-llama3 8b](https://huggingface.co/binxia/llmga-cn-llama3-8b-it-full-finetune), [llmga-cn-gemma 2b](https://huggingface.co/binxia/llmga-cn-gemma-2b-it-full-finetune), and [llmga-cn-qwen2 0.5b](https://huggingface.co/binxia/llmga-cn-Qwen2-0.5B-full-finetune).
- [x] [2024.07.06] We release new version LLMGA's [training datasets](https://huggingface.co/datasets/binxia/LLMGA-datasetv2/tree/main), including texts and images.
- [x] [2024.07.05] The pre-trained model has been released, including [llmga-vicuna 7b](https://huggingface.co/binxia/llmga-vicuna-7b-v1.5-full-finetune/tree/main), [llmga-mistral 7b](https://huggingface.co/binxia/llmga-mistral_instruct-full-finetune/tree/main), [llmga-llama3 8b](https://huggingface.co/binxia/llmga-llama3-8b-it-full-finetune/tree/main), [llmga-vicuna7b](https://huggingface.co/binxia/llmga-vicuna-7b-v1.5-full-finetune/tree/main), [llmga-qwen2 0.5b](https://huggingface.co/binxia/llmga-Qwen2-0.5B-full-finetune/tree/main), [llmga-qwen2 1.5b](https://huggingface.co/binxia/llmga-Qwen2-1.5B-full-finetune/tree/main), [llmga-qwen2 7b](https://huggingface.co/binxia/llmga-Qwen2-7B-full-finetune/tree/main), [llmga-phi3 3b](https://huggingface.co/binxia/llmga-Phi-3-mini-128k-full-finetune/tree/main), and [llmga-gemma 2b](https://huggingface.co/binxia/llmga-gemma-2b-it-full-finetune/tree/main).
- [x] [2024.07.05] The code has been updated.
- [x] [2024.07.04] I am organizing and uploading the new version of the LLMGA code and the dataset and model. I will have a status update when I complete this process, please wait for me for a few days. Notably, in this new version, we build our LLMGA on different base LLM models, such as **Llama2 7b**, **Mistral 7b**, **LLama3 8b**, **Qwen2 0.5b**, **Qwen2 1.5b**, **Qwen2 7b**, **Phi3 3b**, and **gemma 2b**. They have different performance and model sizes, as well as commercial licenses, there is always one that can meet your usage scenario.
      
**Old Version:**
- [x] [2023.12.20]   We release LLMGA's [training datasets].
- [x] [2023.12.20]    We release the gradio codes of LLMGA7b-SDXL-T2I.
- [x] [2023.12.08]   We release LLMGA7b-SDXL-T2I [demo].
- [x] [2023.11.30]   We have released the code for DiffRIR. It can effectively eliminate differences in brightness, contrast, and texture between generated and preserved regions in inpainting and outpainting. Considering its applicability to projects beyond LLMGA, we have open-sourced it at [Github](https://github.com/Zj-BinXia/DiffRIR).
- [x] [2023.11.29]   The models is released at [Huggingface].
- [x] [2023.11.29]   The training and inference code is released.
- [x] [2023.11.29]  We will upload all models, code, and data within a week and further refine this project.
- [x] [2023.11.28]    GitHub repo is created.

---

> **Abstract:** In this paper, we introduce a Multimodal Large Language Model-based Generation Assistant (LLMGA), leveraging the vast reservoir of knowledge and proficiency in reasoning, comprehension, and response inherent in Large Language Models (LLMs) to assist users in image generation and editing. Diverging from existing approaches where Multimodal Large Language Models (MLLMs) generate fixed-size embeddings to control Stable Diffusion (SD), our LLMGA provides a detailed language generation prompt for precise control over SD. This not only augments LLM context understanding but also reduces noise in generation prompts, yields images with more intricate and precise content, and elevates the interpretability of the network. To this end, we curate a comprehensive dataset comprising prompt refinement, similar image generation, inpainting \& outpainting, and instruction-based editing. Moreover, we propose a two-stage training scheme. In the first stage, we train the MLLM to grasp the properties of image generation and editing, enabling it to generate detailed prompts. In the second stage, we optimize SD to align with the MLLM's generation prompts. Additionally, we propose a reference-based restoration network to alleviate texture, brightness, and contrast disparities between generated and preserved regions during inpainting and outpainting. Extensive results show that LLMGA has promising generation and editing capabilities and can enable more flexible and expansive applications in an interactive manner.

---

## Why do you need LLMGA?

- [x] **Generation Assiatant**. As a unified system, LLMGA can generate and edit images using methods such as Text-to-Image (T2I), inpainting, outpainting, and instruction-based editing through conversational interactions with users. By leveraging the extensive knowledge and understanding of image design from LLMGA, users can easily produce and revise images to obtain highly satisfactory images.
- [x]  **Design Expert**. LLMGA incorporates an extensive array of image design data, offering deep insights for a wide range of design tasks, including logo creation, game character design, poster design, T-shirt design, infographic design, and more.
- [x]  **Illustration Generation**. LLMGA can interactively generate story illustrations based on user-input story snippets.
- [x]  **Picture Book Generation**. With a single user's instruction, LLMGA can generate an interwoven storybook of text and illustrations.
- [x]  **Multilingual Support**.Through the multilingual adaptation of the LLMGA, T2I and editing model can generate content using Chinese language instructions.
- [x]  **Flexible Expansion**. LLMGA offers enhanced flexibility by integrating with external plugins like ControlNet, enabling a wider range of functionalities.
- [x] To be continued ......

<div align=center>
<img width="100%" src="imgs/github_poster1.png"/>
</div>

<div align=center>
<img width="100%" src="imgs/demo1.png"/>
</div>

<div align=center>
<img width="100%" src="imgs/demo2.png"/>
</div>


## Contents
- [TODO](#todo)
- [Install](#install)
- [Model](#model)
- [Preparation](#Preparation)
- [Train](#train)
- [Inference](#inference)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## TODO
- [x] Support gradio demo.
- [ ] Support more generation models



## Install
Please follow the instructions below to install the required packages.
1. Clone this repository
```bash
git clone https://github.com/dvlab-research/LLMGA.git
```

2. Install Package
```bash
conda create -n llmga python=3.9 -y
conda activate llmga
cd LLMGA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
cd ./llmga/diffusers
pip install . 
```

3. Install additional packages for training cases
```bash
pip install -e ".[train]"
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install datasets
pip install albumentations
pip install ninja
```



## Model


<div align=center>
<img width="100%" src="imgs/method.png"/>
</div>



## Preparation

### Training Dataset
We provide the training data for LLMGA training. 

please download [LLMGA datasets](https://huggingface.co/datasets/binxia/LLMGA-datasetv2) and [LLaVA pretrain datasets](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). 

Besides, download LLaVA1.5 instruction tuning datasets [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). 

Please organize these downloaded data as in [Structure](#structure).

### The MLP Projector Pretrained Weights
We recommend users to download the [pretrained MLP projector weights](https://huggingface.co/binxia/LLMGA-pretrained-mlp/tree/main). Then put them in `./checkpoints` following [Structure](#structure).


### Inference Pretrained Weights

Please download MLLM Models and SD models from the following links. For example, you can download [LLMGA-MLLM7b](https://huggingface.co/binxia/llmga-llama-2-7b-chat-full-finetune) and [LLMGA-SDXL-T2I](https://huggingface.co/binxia/llmga-sdxl-t2i) to realize LLMGA7b-T2I functionality. Please organize them as in [Structure](#structure).


<table>
  <tr>
    <th align="left">MLLM Model (support English)</th>
    <th align="center">Pretrained Models</th>
  </tr>
  <tr>
    <td align="left">llmga-vicuna 7b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-vicuna-7b-v1.5-full-finetune/tree/main">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-mistral 7b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-mistral_instruct-full-finetune/tree/main">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-llama3 8b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-llama3-8b-it-full-finetune/tree/main">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-qwen2 0.5b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-Qwen2-0.5B-full-finetune/tree/main">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-qwen2 1.5b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-Qwen2-1.5B-full-finetune/tree/main">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-qwen2 7b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-Qwen2-7B-full-finetune/tree/main">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-phi3 3b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-Phi-3-mini-128k-full-finetune/tree/main">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-gemma 2b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-gemma-2b-it-full-finetune/tree/main">Download</a></td>
  </tr>
</table>

<table>
  <tr>
    <th align="left">MLLM Model (further support Chinese and English)</th>
    <th align="center">Pretrained Models</th>
  </tr>
  <tr>
    <td align="left">llmga-cn-vicuna 7b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-cn-vicuna-7b-v1.5-full-finetune">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-cn-llama3 8b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-cn-llama3-8b-it-full-finetune">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-cn-gemma 2b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-cn-gemma-2b-it-full-finetune">Download</a></td>
  </tr>
  <tr>
    <td align="left">llmga-cn-qwen2 0.5b</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-cn-Qwen2-0.5B-full-finetune">Download</a></td>
  </tr>
</table>


<table>
  <tr>
    <th align="left">SD Model</th>
    <th align="center">Pretrained Models</th>
  </tr>
  <tr>
    <td align="left">LLMGA-SD15-T2I</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-sd15-t2i-v2">Download</a></td>
  </tr>
  <tr>
    <td align="left">LLMGA-SD15-Inpainting</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-sd15-inpainting-v2">Download</a></td>
  </tr>
  <tr>
    <td align="left">LLMGA-SDXL-T2I</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-sdxl-t2i">Download</a></td>
  </tr>
  <tr>
    <td align="left">LLMGA-SDXL-Inpainting</td>
    <td align="center"><a href="https://huggingface.co/binxia/llmga-sdxl-inpainting-v2">Download</a></td>
  </tr>
</table>

### Structure

The folder structure should be organized as follows before training.

```
LLMGA
├── llmga
├── scripts
├── work_dirs
├── checkpoints
│   ├── llmga-Phi-3-mini-128k-pretrain
│   ├── llmga-Qwen2-0.5B-pretrain
│   ├── llmga-llama3-8b-pretrain
│   ├── llmga-mistral-pretrain
│   ├── llmga-vicuna-7b-v1.5-pretrain
│   ├── llmga-Phi-3-mini-128k-full-finetune
│   ├── llmga-Qwen2-0.5B-full-finetune
│   ├── llmga-llama3-8b-it-full-finetune
│   ├── llmga-mistral_instruct-full-finetune
│   ├── llmga-vicuna-7b-v1.5-full-finetune
│   ├── llmga-cn-vicuna-7b-v1.5-full-finetune
│   ├── llmga-cn-Qwen2-0.5B-full-finetune
│   ├── llmga-sdxl-t2i
│   ├── llmga-sd15-inpainting-v2
│   ├── llmga-sd15-t2i-v2
├── data
│   │── jsons
│   │   ├── llmga-data
│   │   │   ├── Edit/train.json
│   │   │   ├── inpainting/train.json
│   │   │   ├── SG/train.json
│   │   │   ├── T2I/train.json
│   │   ├── text-data
│   │   │   ├── alpaca_gpt4_sharegpt_en_clean2.json
│   │   │   ├── lima.json
│   │   │   ├── oasst2.json
│   │   ├── llava_v1_5_mix665k.json
│   ├── llmga-imgs
│   │   ├── COCO
│   │   ├── LAION
│   │   ├── JourneyDB
│   ├── llava_pretrain
│   │   ├──images
│   ├── llava-imgs
│   │   ├── coco
│   │   │   ├── train2017
│   │   ├── gqa
│   │   │   ├── images
│   │   ├── ocr_vqa
│   │   │   ├── images
│   │   ├── textvqa
│   │   │   ├── train_images
│   │   ├── vg
│   │   │   ├── VG_100K
│   │   │   ├── VG_100K_2
```



## Train

LLMGA is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

Please make sure you download and organize the data following [Preparation](#Preparation) before training. Here, we just take training llmga vicuna 7b as an example. For more model training scripts, please check the **./scripts** folder.  

### Pretrain

```bash
bash scripts/pretrain_vicuna_7b.sh
```

### First Stage Training

```bash
bash scripts/train_llmga_s1_7b_vicuna.sh
```

### Second Stage Training

train LLMGA based on SD1.5-T2I
```bash
bash scripts/train_llmga_s2_sd15_t2i.sh
```

train LLMGA based on SD1.5-Inpainting
```bash
bash scripts/train_llmga_s2_sd15_inpaint.sh
```

## Inference

### CLI Inference

Use LLMGA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization.
Here, we just give some examples for T2I, inpainting and instruction-based editing. For more model inference scripts, please check the **./scripts** folder. 

For **T2I** generation task.
```bash
bash scripts/test-llmga-sdxl-t2i.sh
```

For **inpainting or outpainting** task.
```bash
bash scripts/test-llmga-sd15-inpainting.sh
```

For **instruction based editing** task.
```bash
bash scripts/test-llmga-sd15-editing.sh
```

### Gradio Inference
```bash
bash scripts/run_gradio_t2i.sh
```






## Citation
If you find this repo useful for your research, please consider citing the paper
```
@article{xia2023llmga,
  title={LLMGA: Multimodal Large Language Model based Generation Assistant},
  author={Xia, Bin and Wang, Shiyin, and Tao, Yingfan and Wang, Yitong and Jia, Jiaya},
  journal={ECCV},
  year={2024}
}
```

## Acknowledgement
We would like to thank the following repos for their great work:

- This work utilizes MLLM from [LLaVA](https://github.com/haotian-liu/LLaVA).
- This work utilizes Stable Diffusion from [diffusers](https://github.com/huggingface/diffusers).



