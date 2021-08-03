import numpy as np
import pickle

def load_dataset(filename):
    file = open(filename,'rb')
    data = pickle.load(file)
    file.close()
    x, y = data[0], data[1]
    x = np.array(x)
    y = np.array(y)
    return (x, y)

