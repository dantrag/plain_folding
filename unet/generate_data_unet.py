import sys
import random

import numpy as np
import pickle

from folding import performe_folding

size = 10000
seed = "input/unfolded_real_mask_padded.png"
if len(sys.argv) > 1:
    size = int(sys.argv[1])

folded_images = performe_folding(seed, max_folds=2, num_examples=size)
print("Image shape", end='')
print(folded_images[0].shape)

random.shuffle(folded_images)
x_data = []
y_data = []
for y in folded_images:
    x = y.astype('float32').copy()
    x[x > 0] = 1
    x_data.append(np.expand_dims(x, axis=-1))
    y_data.append(np.expand_dims(y.astype('float32') / 160, axis=-1))

validation = int(size * 0.1)
test = int(size * 0.1)
train = size - validation - test

#save the dataset
with open("dataset_train_unet.pkl", 'wb') as f:
    pickle.dump((x_data[:train], y_data[:train]), f)

with open("dataset_validation_unet.pkl", 'wb') as f:
    pickle.dump((x_data[train:train+validation], y_data[train:train+validation]), f)

with open("dataset_test_unet.pkl", 'wb') as f:
    pickle.dump((x_data[-test:], y_data[-test:]), f)

