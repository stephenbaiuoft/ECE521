import numpy as np
import tensorflow as tf
from e122 import *

def main():
    knn(2)

def knn(k):
    l = datagen()
    trainData = l[0]
    trainTarget = l[1]
    vData = l[2]
    vTarget = l[3]
    testData = l[4]
    testTarget = l[5]

    x1_ = tf.placeholder('float32')
    z1_ = tf.placeholder('float32')
    D_ = D(x1_, z1_)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    testData = np.array([[1],[2],[3]])
    trainData = np.array([[1],[2],[10],[30]])

   # l = sess.run(D_, feed_dict={x1_: tData, z1_: testData})
   # print(l)

    dis = sess.run(D_, feed_dict={x1_: testData, z1_: trainData})
    dis = dis * -1
    print(dis)
    #sess.run(dis)

    res = tf.nn.top_k(dis,k)

    print("\n\n*************\n")
    print( sess.run(res.indices))
    print("\n\n*************\n")
    #print (trainData.shape[0])
    #print (testData.shape[0])
    matrix = np.zeros( [testData.shape[0],
                      trainData.shape[0]])

# [ [0 1]
#   [1 0]
#   [1 0] ]
    print (testData.item( 1, sess.run (res.indices[0])) )

    print(matrix)
   # sess.run(res)


def datagen():
    np.random.seed(521)
    Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
             + 0.5 * np.random.randn(100 , 1)
    # randIdx from 0,1,...99
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    return (trainData, trainTarget,validData, validTarget,testData,testTarget)


main()