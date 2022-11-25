"""
Run the stimuli used in the MEG experiment through the model and record the
activity in each layer.
"""
import torch
from torchvision import transforms
import numpy as np
import pickle
import pandas as pd
#import mkl
#mkl.set_num_threads(4)
from PIL import Image
from tqdm import tqdm

import network
import dataloader

data_path = '/tmp'

classes = dataloader.WebDataset('/m/nbe/scratch/reading_models/datasets/epasana-1kwords').classes
classes.append(pd.Series(['noise'], index=[1000]))

# In order to get word2vec vectors, the KORAANI class was replaced with
# KORAANIN. Switch this back, otherwise, this word will be erroneously flagged
# as being misclassified.
classes[classes == 'KORAANIN'] = 'KORAANI'

# Load the TIFF images presented in the MEG experiment and apply the
# ImageNet preprocessing transformation to them.
stimuli = pd.read_csv('/tmp/stimuli.csv')
preproc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
images = []
for fname in tqdm(stimuli['tif_file'], desc='Reading images'):
    with Image.open(f'/tmp/stimulus_images/{fname}') as orig:
        image = Image.new('RGB', (224, 224), '#696969')
        image.paste(orig, (12, 62))
        image = preproc(image).unsqueeze(0)
        images.append(image)
images = torch.cat(images, 0)

# Load the model and feed through the images. Make sure to feed the images
# through as a single batch, because I don't trust the BatchNormalization2d
# layers to behave predictably otherwise.
checkpoint = torch.load('model_best.pth.tar', map_location='cpu')
model = network.VGG11.from_checkpoint(checkpoint, freeze=True)

layer_outputs = model.get_layer_activations(
    images,
    feature_layers=[2, 6, 13, 20, 27],
    classifier_layers=[1, 4, 6]
)

layer_activity = []
for output in layer_outputs:
    if output.shape[-1] == 2001:
        print('Removing nontext class')
        output = np.hstack((output[:, :2000], output[:, 2001:]))
        print('New output shape:', output.shape)
    layer_activity.append(output)

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

mean_activity = np.array([np.square(a.reshape(len(stimuli), -1)).mean(axis=1)
                          for a in layer_activity])

with open('../data/models/stimuli_model_layer_activity.pkl', 'wb') as f:
    pickle.dump(dict(layer_names=layer_names, mean_activity=mean_activity), f)

# Translate output of the model to text predictions
predictions = stimuli.copy()
predictions['predicted_text'] = classes[layer_activity[-1].argmax(axis=1)].values
predictions['predicted_class'] = layer_activity[-1].argmax(axis=1)
predictions.to_csv('predictions.csv')

# How many of the word stimuli did we classify correctly?
word_predictions = predictions.query('type=="word"')
n_correct = (word_predictions['text'] == word_predictions['predicted_text']).sum()
accuracy = n_correct / len(word_predictions)
print(f'Word prediction accuracy: {n_correct}/{len(word_predictions)} = {accuracy * 100:.1f}%')  # 113/118, not bad at all!
