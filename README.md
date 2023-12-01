


<p align="center" width="10%">
<img src="imgs/logo.png" style="width: 30%" align=center> 
</p>


# LLMGA: Multimodal Large Language Model based Generation Assistant

[Bin Xia](https://scholar.google.com/citations?user=rh2fID8AAAAJ&hl=zh-CN), [Shiyin Wang](), [Yingfan Tao](https://scholar.google.com/citations?user=GYDnPdQAAAAJ&hl=zh-CN&oi=ao), [Yitong Wang](https://scholar.google.com/citations?user=NfFTKfYAAAAJ&hl=zh-CN), and [Jiaya Jia](https://scholar.google.com/citations?user=XPAkzTEAAAAJ&hl=zh-CN&oi=ao)

<a href="https://llmga.github.io/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href='https://llmga.github.io/'><img src='https://img.shields.io/badge/Project-Demo-violet'></a>
<a href="https://arxiv.org/pdf/2311.16500.pdf"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://huggingface.co/binxia'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/binxia'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

## News
- [x] [2023.11.30]   We have released the code for DiffRIR. It can effectively eliminate differences in brightness, contrast, and texture between generated and preserved regions in inpainting and outpainting. Considering its applicability to projects beyond LLMGA, we have open-sourced it at [Github](https://github.com/Zj-BinXia/DiffRIR).
- [x] [2023.11.29]   🔥 The models is released at [Huggingface](https://huggingface.co/binxia).
- [x] [2023.11.29]   🔥 The training and inference code is released.
- [x] [2023.11.29]  We will upload all models, code, and data within a week and further refine this project.
- [x] [2023.11.28]   🔥 GitHub repo is created.

---

> **Abstract:** In this paper, we introduce a Multimodal Large Language Model-based Generation Assistant (LLMGA), leveraging the vast reservoir of knowledge and proficiency in reasoning, comprehension, and response inherent in Large Language Models (LLMs) to assist users in image generation and editing. Diverging from existing approaches where Multimodal Large Language Models (MLLMs) generate fixed-size embeddings to control Stable Diffusion (SD), our LLMGA provides a detailed language generation prompt for precise control over SD. This not only augments LLM context understanding but also reduces noise in generation prompts, yields images with more intricate and precise content, and elevates the interpretability of the network. To this end, we curate a comprehensive dataset comprising prompt refinement, similar image generation, inpainting $\&$ outpainting, and visual question answering. Moreover, we propose a two-stage training scheme. In the first stage, we train the MLLM to grasp the properties of image generation and editing, enabling it to generate detailed prompts. In the second stage, we optimize SD to align with the MLLM's generation prompts. Additionally, we propose a reference-based restoration network to alleviate texture, brightness, and contrast disparities between generated and preserved regions during image editing. Extensive results show that LLMGA has promising generative capabilities and can enable wider applications in an interactive manner. 

## Contents
- [Demo](#demo)
- [Install](#install)
- [Model](#model)
- [Preparation](#preparation)
- [Train](#train)
- [Inference](#inference)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Demo
We provide some selected examples in this section. More examples can be found in our [project page](https://llmga.github.io/). Feel free to try our online [demo](https://llmga.github.io/)!

<div align=center>
<img width="100%" src="imgs/github_poster1.png"/>
</div>

<div align=center>
<img width="100%" src="imgs/github_poster2.png"/>
</div>

## Install
Please follow the instructions below to install the required packages.
1. Clone this repository
```bash
git clone https://github.com/Zj-BinXia/LLMGA.git
```

2. Install Package
```bash
conda create -n llmga python=3.9 -y
conda activate llmga
cd 
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
cd ./llmga/diffusers
pip install . 
```

3. Install additional packages for training cases
```bash
pip install datasets
pip install albumentations
pip install ninja
pip install flash-attn --no-build-isolation
```

## Model


<div align=center>
<img width="100%" src="imgs/method.png"/>
</div>

## Preparation

## Train

## Inference

## TODO
- [ ]

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@article{xia2023llmga,
  title={LLMGA: Multimodal Large Language Model based Generation Assistant},
  author={Xia, Bin and Wang, Shiyin, and Tao, Yingfan and Wang, Yitong and Jia, Jiaya},
  journal={arXiv preprint arXiv:2311.16500},
  year={2023}
}
```

## Acknowledgement
We would like to thank the following repos for their great work:

- This work utilizes MLLM from [LLaVA](https://github.com/haotian-liu/LLaVA).
- This work utilizes Stable Diffusion from [diffusers](https://github.com/huggingface/diffusers).



