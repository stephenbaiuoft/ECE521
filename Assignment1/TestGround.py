# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]

import tensorflow as tf
import numpy as np


class myClass(object):

    def __init__(self):
        self.input = None
        self.sess = tf.InteractiveSession()

    def slice(self):
        self.input = [[[1, 1, 1], [2, 2, 2]],
                  [[3, 3, 3], [4, 4, 4]],
                  [[5, 5, 5], [6, 6, 6]]]

        print(self.sess.run(tf.slice(input, [1, 0, 0], [1, 1, 3])))

if __name__ == "__main__":
    print("Testing Space\n")

