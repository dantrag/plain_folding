import sys
import random

import numpy as np
import tensorflow
from tensorflow.python.keras import backend as K
from cv2 import *
#from PIL import Image

from model_unet import unet
from utils import load_dataset

def integrate(image, factor):
    image = image * factor
    image.round()
    return np.sum(image)

def normalize_image(image, truth):
    image = image.astype('int')
    true_integral = np.sum(truth)
    min_factor = 0
    max_factor = 16
    while max_factor - min_factor > 0.0001:
        factor = (max_factor + min_factor) / 2
        integral = integrate(image, factor)
        if integral >= true_integral:
            max_factor = factor
        else:
            min_factor = factor
    image = np.round(image * factor)
    image = image.astype('uint8')
    return image

dataset_name = "dataset_train_unet.pkl"
validate_name = "dataset_validate_unet.pkl"
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        validate_name = sys.argv[2]

model = unet()
#odel_checkpoint = ModelCheckpoint('checkpoints.hdf5', monitor='loss', verbose=1, save_best_only=True)

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = tensorflow.compat.v1.Session(config=config)
K.set_session(session)

x, y = load_dataset(dataset_name)
batch_size = 2
epoch_count = 10

xval, yval = load_dataset(validate_name)

print("Training started", flush=True)

for epoch in range(epoch_count):
    print("epoch %d..." % epoch, end='', flush=True)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    x_batches = [x[start:start + batch_size] for start in range(0, len(x), batch_size)]
    y_batches = [y[start:start + batch_size] for start in range(0, len(y), batch_size)]

    iteration_count = len(x_batches)
    for i in range(iteration_count):
        model.train_on_batch(np.array(x_batches[i]), np.array(y_batches[i]))

    print("done", flush=True)

    results = model.evaluate(x, y, batch_size=1, verbose=0)
    print(model.metrics_names, end=' ')
    print(results, flush=True)

    yhat = model.predict(np.array(x[:10]), batch_size=1)
    #yhat *= 16
    #yhat = np.around(yhat).astype('uint8')
    #yhat *= 10

    prediction = []
    for i in range(20):
        image = None
        if i & 1:
            image = yhat[i // 2] * 16
        else:
            image = y[i // 2] * 16
        image = np.squeeze(image, axis=2)
        if i & 1:
            image = normalize_image(image, y[i // 2] * 16)
        else:
            image = image.astype('uint8')
        image *= 10
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        bordersize = 5
        border = cv2.copyMakeBorder(image,
                                    top=bordersize,
                                    bottom=bordersize,
                                    left=bordersize,
                                    right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])
        prediction.append(border)

    rows = []
    for i in range(0, 20, 2):
        t_row = prediction[i:i+2]
        x_row = np.concatenate([t_row[i] for i in range(2)], axis=1)
        rows.append(x_row)
        data_example = np.concatenate([rows[i] for i in range(len(rows))], axis=0)

    cv2.imwrite("validation_epoch%d.png" % epoch, data_example)

model.save('unet_trained.h5')

print("Training finished")
