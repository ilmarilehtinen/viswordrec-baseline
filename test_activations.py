import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import pickle
import pandas as pd
from PIL import Image

import dataloader
import network
import os

data_path = '../data'
local_datapath = '/tmp'
imagenet_datapath = '/m/nbe/scratch/reading_models/datasets/imagenet-1k'

epasana_dataset = dataloader.WebDataset(os.path.join(local_datapath, 'epasana-1kwords'), train=False)
imagenet_dataset = dataloader.WebDataset(imagenet_datapath, train=False)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

imagenet_images = []
epasana_images = []

n_images = 100
for i in range(n_images):
    img, label = epasana_dataset.__next__()
    img = transform(img).unsqueeze(0)
    epasana_images.append(img)    

    img, label = imagenet_dataset.__next__()
    img = transform(img).unsqueeze(0)
    imagenet_images.append(img)

epasana_images = torch.cat(epasana_images, 0)
imagenet_images = torch.cat(imagenet_images, 0)
    
checkpoint = torch.load(f'/m/nbe/work/lehtini6/viswordrec-baseline/model_best.pth.tar')
model = network.VGG11.from_checkpoint(checkpoint, freeze=True)

epasana_outputs = model.get_layer_activations(
    epasana_images,
    feature_layers=[2,6,13,20,27],
    classifier_layers=[1,4,6]
)

imagenet_outputs = model.get_layer_activations(
    imagenet_images,
    feature_layers=[2,6,13,20,27],
    classifier_layers=[1,4,6]
)


epasana_layer_activity = [output for output in epasana_outputs]
imagenet_layer_activity = [output for output in imagenet_outputs]

layer_names = [
    'conv1_relu',
    'conv2_relu',
    'conv3_relu',
    'conv4_relu',
    'conv5_relu',
    'fc1_relu',
    'fc2_relu',
    'word_relu',
]

for fmap in epasana_layer_activity:
    print(fmap.shape)

#epasana_mean_activity = np.array([np.square(a.reshape(n_images, -1)).mean(axis=1)
#                          for a in epasana_layer_activity])

#imagenet_mean_activity = np.array([np.square(a.reshape(n_images, -1)).mean(axis=1) for a in imagenet_layer_activity])

with open(f'{data_path}/epasana_model_layer_activity.pkl', 'wb') as f:
    pickle.dump(dict(layer_names=layer_names, mean_activity=epasana_layer_activity), f)

with open(f'{data_path}/imagenet_model_layer_activity.pkl', 'wb') as f:
    pickle.dump(dict(layer_names=layer_names, mean_activity=imagenet_layer_activity), f)
