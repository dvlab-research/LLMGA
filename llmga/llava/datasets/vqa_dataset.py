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
from llmga.llava.datasets.utils import preprocess_multimodal,preprocess
# import spacy
# import pytextrank

from torchvision import transforms
from torch.utils.data import ConcatDataset
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class VQADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(VQADataset, self).__init__()
        self.data_args = data_args
        list_vqa_dict = json.load(open(data_path, "r"))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_vqa_dict = list_vqa_dict
        self.len_vqa = len(self.list_vqa_dict)



    def __len__(self):
        return self.len_vqa


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        tp = self.list_vqa_dict[i].copy()
        image_folder = self.data_args.image_folder

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
