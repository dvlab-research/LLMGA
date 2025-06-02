import glob
import logging
import os
import random

import cv2
import io
import numpy as np
import hashlib
import lmdb
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop
from PIL import Image
from enum import Enum
import csv
import pandas as pd
import json




class LoadImageFromLmdb(object):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.txn = None

    def __call__(self, key):
        if self.txn is None:
            env = lmdb.open(self.lmdb_path, max_readers=4,
                            readonly=True, lock=False,
                            readahead=True, meminit=False)
            self.txn = env.begin(write=False)
        image_buf = self.txn.get(key.encode())
        with Image.open(io.BytesIO(image_buf)) as image:
            if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                image = image.convert("RGBA")
                white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                white.paste(image, mask=image.split()[3])
                image = white
            else:
                image = image.convert("RGB")
        return image





class Text2ImgTrainDataset(Dataset):
    def __init__(self, indir, args=None):
        json_path = "./data/jsons/llmga-data/T2I/train.json"
        self.prompt_dict = json.load(open(json_path,"r"))
        
        self.image_folder = "./data/imgs"
        
        self.train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution), #if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

        self.len=len(self.prompt_dict)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        prompt = self.prompt_dict[index]["caption"]
        image_path=os.path.join(self.image_folder,self.prompt_dict[index]["image"])
        img=Image.open(image_path)
        img=self.train_transforms(img)

        res = {
                "pixel_values": img,
                "caption": prompt,
                }
        return res






