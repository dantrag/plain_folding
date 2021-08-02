import sys

import numpy as np
from cv2 import *
from PIL import Image
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.python.keras import backend as K

from utils import load_dataset

def integrate(image, factor):
    image = image * factor
    image.round()
    return np.sum(image)

def postprocess(image, truth):
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

model_name = "unet_trained.h5"
dataset_name = "dataset_validate_unet.pkl"
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = tensorflow.compat.v1.Session(config=config)
K.set_session(session)

model = tensorflow.keras.models.load_model(model_name)
x, y = load_dataset(dataset_name)
results = model.evaluate(x, y, batch_size=1)
print(model.metrics_names)
print(results)

y *= 16
y = y.astype('uint8')
y *= 10

yhat = model.predict(np.array(x[:10]), batch_size=1)
yhat *= 16
yhat = np.around(yhat).astype('uint8')
yhat *= 10

prediction = []
for i in range(20):
    image = None
    if i & 1:
        image = yhat[i // 2]
    else:
        image = y[i // 2]
    image = np.squeeze(image, axis=2)
    if i & 1:
        image = postprocess(image, y[i // 2])
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
    t_row=prediction[i:i+2]
    x = np.concatenate([t_row[y] for y in range(2)], axis=1)
    rows.append(x)
    data_example = np.concatenate([rows[y] for y in range(len(rows))], axis=0)

cv2.imwrite("validation.png", data_example)

