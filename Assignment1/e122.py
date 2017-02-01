import tensorflow as tf
import numpy as np



def D(x, z):
    y = tf.squared_difference(x[:, tf.newaxis], z)
    result = tf.reduce_sum(y, 2)
    return result

def main():
    x1_ = tf.placeholder('float32')
    z1_ = tf.placeholder('float32')
    D_ = D(x1_, z1_)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    x1 = np.array([[1],[2],[3]])
    z2 = np.array([[1],[2]])

    testData = np.array([[1],[10],[20],[30],[40]])
    tData = np.array([[3],[17]])

    l = sess.run(D_, feed_dict={x1_: x1, z1_: z2})
    print("hmm should not print htis")
    print(l)
    print("\n\n\n")

if __name__ == "__main__":
    main()