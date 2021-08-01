import sys

import numpy as np

from model_unet import unet
from utils import load_dataset

dataset_name = "dataset_train_unet.pkl"
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]

model = unet()
#odel_checkpoint = ModelCheckpoint('checkpoints.hdf5', monitor='loss', verbose=1, save_best_only=True)

x, y = load_dataset(dataset_name)
batch_size = 2
x_batches = [x[start:start + batch_size] for start in range(0, len(x), batch_size)]
y_batches = [y[start:start + batch_size] for start in range(0, len(y), batch_size)]

epochs = len(x_batches)
print("Training started")
for i in range(epochs):
    if i % 10 == 0:
        print("epoch %d" % i)
    model.train_on_batch(np.array(x_batches[i]), np.array(y_batches[i]))
model.save('unet_trained.h5')
print("Training finished")
