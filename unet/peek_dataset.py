import sys
import random

import pickle
import numpy as np

#from model_unet import *
#from train_unet import load_dataset
from PIL import Image

def load_dataset(filename):
    file = open(filename,'rb')
    data = pickle.load(file)
    file.close()
    x, y = data[0], data[1]
    x = np.array(x)
    y = np.array(y)
    return (x, y)

dataset_name = sys.argv[1]
x, y = load_dataset(dataset_name)

result = Image.fromarray(np.around(np.squeeze(random.choice(y), axis=2) * 160).astype('uint8'), 'L')
result.show()

