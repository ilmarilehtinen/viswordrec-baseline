import torch
from torchvision import transforms
import numpy as np
import pickle
import pandas as pd
from PIL import Image

import dataloader
import network
import os


datapath = '/tmp'

dataset = dataloader.WebDataset(os.path.join(datapath, 'epasana-1kwords'), train=False)

for img, label in dataset:
    print(label)
    im = Image.open(img)
    