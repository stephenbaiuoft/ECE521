import numpy as np
import tensorflow as tf

class model:
    def __init__(self):
        with np.load("tinymnist.npz") as data :
            trainData, trainTarget = data ["x"], data["y"]
            validData, validTarget = data ["x_valid"], data ["y_valid"]
            testData, testTarget = data ["x_test"], data ["y_test"]

    def run(self):
        pass

#print(y)
#print(np.shape(y))