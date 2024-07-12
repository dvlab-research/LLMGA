import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llmga.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_OUTPUT_START_TOKEN, DEFAULT_OUTPUT_END_TOKEN
from torch.utils.data import Dataset
from llmga.llava.train.llava_trainer import LLaVATrainer

from llmga.llava import conversation as conversation_lib
from llmga.llava.model import *
from llmga.llava.mm_utils import tokenizer_image_token

from PIL import Image
from llmga.llava.masks.make_mask import get_mask_generator
from llmga.llava.prompt_temp import outpaint_prompt, inpaint_prompt, textextend_prompt, textextend_prompt_behind, regen_prompt, ques_prompt, \
    textextend_prompt2
import random
import numpy as np

import io
# import spacy
# import pytextrank

from torchvision import transforms
from torch.utils.data import ConcatDataset
from llmga.llava.datasets.utils import preprocess_multimodal,preprocess
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class TextDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                data_path2: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(TextDataset, self).__init__()
        self.data_args = data_args
        self.list_alpaca_dict = json.load(open(os.path.join(data_path,"alpaca_gpt4_sharegpt_en_clean2.json"), "r"))
        self.len_alpaca = len(self.list_alpaca_dict)
        self.list_lima_dict = json.load(open(os.path.join(data_path,"lima.json"), "r"))
        self.len_lima = len(self.list_lima_dict)
        self.list_oasst2_dict = json.load(open(os.path.join(data_path,"oasst2.json"), "r"))
        self.len_oasst2 = len(self.list_oasst2_dict)
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        list_t2i_dict = json.load(open(data_path2, "r"))
        self.list_t2i_dict = list_t2i_dict
        self.len_t2i = len(self.list_t2i_dict)

    def __len__(self):
        return  self.len_lima + self.len_oasst2 + self.len_t2i

    def get_dataset_alpaca(self,i):
        if i<self.len_lima:
            data = self.list_lima_dict[i]
        elif i<self.len_lima+self.len_oasst2:
            data = self.list_oasst2_dict[i-self.len_lima]
                     
        return data

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i<self.len_t2i:
            tp = self.list_t2i_dict[i].copy()
            text_complete = tp['ori_caption'] 
            conversations = [{"from": "human", "value": random.choice(ques_prompt)},
                {"from": "gpt", "value": tp["caption"]}]
    
            tp = { "conversations": conversations}

            if random.random() < 0.5:  # description
                tp_prompt = textextend_prompt2
                is_generation = False
            else:  # generation
                tp_prompt = textextend_prompt
                is_generation = True
            
            is_behind=random.random() < 0.5
            if is_generation and is_behind:
                tp_prompt = textextend_prompt_behind

            if is_behind:
                tp["conversations"][0]["value"] = text_complete + " " + random.choice(tp_prompt)
            else:
                tp["conversations"][0]["value"] = random.choice(tp_prompt) + " " + text_complete

            if is_generation:
                tp["conversations"][1]["value"] = DEFAULT_OUTPUT_START_TOKEN + " " + tp["conversations"][1][
                    "value"] + " " + DEFAULT_OUTPUT_END_TOKEN
        else:
            tp=self.get_dataset_alpaca(i-self.len_t2i)

        if isinstance(i, int):
            sources = [tp]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = tp['image']
            processor = self.data_args.image_processor

            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

            def resize2square(pil_img):
                result = pil_img.resize((512, 512), Image.ANTIALIAS)
                return result

            def cropresize(pil_img):
                width, height = pil_img.size
                if width > height:
                    result = pil_img.resize((int(width / height * 512), 512), Image.ANTIALIAS)
                else:
                    result = pil_img.resize((512, int(height / width * 512)), Image.ANTIALIAS)
                result = transforms.CenterCrop(512)(result)
                return result

            if "laion" in tp['image']:
                image = cropresize(image)
            else:
                image = resize2square(image)
            
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in tp))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if 'image' in tp:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
