import sys

import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import *

from utils import load_dataset

dataset_name = "dataset_validation_unet.pkl"
model_name = "unet_trained.h5"
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]

model = tensorflow.keras.models.load_model(model_name)
x, y = load_dataset(dataset_name)
results = model.test_on_batch(x, y)

print(model.metrics_names)
print(results)

