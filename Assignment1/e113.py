import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from e122 import *

def main():
    sess = tf.InteractiveSession()
    l = datagen()
    trainData = l[0]
    trainTarget = l[1]
    validationData = l[2]
    validationTarget = l[3]
    testData = l[4]
    testTarget = l[5]
    Data = l[6]
    Target = l[7]
    k = 3

    #k_validation = get_opt_k(trainData, validationData, trainTarget, validationTarget)
    k_validation = get_opt_k(trainData, trainData, trainTarget, trainTarget)
    #k_test =  get_opt_k(trainData, testData, trainTarget, testTarget)
    #knn(2, trainData, testData)
    #yexpected = predict(trainData, np.array([[1]]), trainTarget, testTarget, k)[1]
    #print("yexpected is:\n", sess.run(yexpected))

    #y_expected = predict(trainData, testData, trainTarget, testTarget, 50)[1]
    #print("MainL y_expected: ", sess.run(y_expected))

    #draw(Data, Target, trainData, testData, trainTarget, testTarget, 5)

def get_opt_k(trainData, testData, trainTarget, testTarget):
    sess = tf.InteractiveSession()

    mse_result = []
    y_result = []
    k_min = 1
    min_mse_loss = sess.run(predict(trainData, np.array([[1]]), trainTarget, testTarget, 1)[0])
    y_expected = sess.run(predict(trainData, np.array([[1]]), trainTarget, testTarget, 1)[1])
    y_result.append(y_expected)

    mse_result.append(min_mse_loss.tolist())

    for k in range(5, 50):
        mse_loss = sess.run(predict(trainData, np.array([[1]]), trainTarget, testTarget, k)[0])
        y_expected = sess.run(predict(trainData, np.array([[1]]), trainTarget, testTarget, k)[1])
        y_result.append(y_expected)
        mse_result.append(mse_loss.tolist())

        if mse_loss < min_mse_loss:
            min_mse_loss = mse_loss
            k_min = k
    print("min_k value is: ", k_min, "with mse_min of :", min_mse_loss)
    print("min_matrix for k 1, 3, 5, 50:\n", mse_result)
    print("y_result values for k 1, 3, 5, 50:\n", y_result)

    return k

def draw(Data, Target, trainData, testData, trainTarget, testTarget, k):
    sess = tf.InteractiveSession()
    X = np.linspace(0.0, 11.0, num=100)[:, np.newaxis]
    Y = []
    #tf.placeholder('float32')
    for x in X:
        testData = np.array([x])
        ysub = sess.run(predict(trainData, testData, trainTarget, testTarget, k)[1])
        #ysub = predict(trainData, testData, trainTarget, testTarget)[1]
        #print("ysub is:", type(ysub))
        #print("ylist is: ", ysub.tolist())
        Y.extend(ysub.tolist())
    #print("Y is ", type(Y))
    #print("Y\n", Y)
    plt.plot(X, Y, 'r--', Data, Target, 'bs')
    plt.show()



def predict(trainData, testData, trainTarget, testTarget, k):
    sess = tf.InteractiveSession()

    # Testing Space
    # testData = np.array([[0], [5], [3]])
    # testTarget = np.array([[2], [4], [6]])
    #
    # trainData = np.array([[1], [2], [10], [30]])
    # trainTarget = np.array([[1.0], [2.0], [10.0], [30.0]])
    # Expected 3.5 and 21.0
    # Got
    # mse value:  3.5 ssd value:  21.0
    knearest = knn(k, trainData, testData)
    #print("k_nearest is:\n", sess.run(knearest))

    yexpected = tf.matmul(knearest, trainTarget)
    #print( "knearest: ", knearest, "trainTarget: ", trainTarget)
    #print("predict y_expected is:\n", sess.run(yexpected))

    sd = tf.squared_difference(yexpected, testTarget)
    #print("sd:\n", sess.run(sd))

    ssd = tf.reduce_sum(sd)
    mse_loss = tf.divide(ssd, 2 * testTarget.shape[0])

    #print("mse value: ", sess.run(mse_loss),"ssd value: ", sess.run(ssd))
    return (mse_loss, yexpected)

    #print("tf shape:", (sess.run(yexpected)))


def knn(k, trainData, testData):
    x1_ = tf.placeholder('float32')
    z1_ = tf.placeholder('float32')
    D_ = D(x1_, z1_)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    #testData = np.array([[1], [2], [3]])
    #trainData = np.array([[1], [2], [10], [30]])

   # l = sess.run(D_, feed_dict={x1_: tData, z1_: testData})
   # print(l)

    dis = sess.run(D_, feed_dict={x1_: testData, z1_: trainData})
    dis *= -1
    #print(dis)

    #sess.run(dis)

    res = tf.nn.top_k(dis, k)
    #print("res.indices:\n", sess.run(res.indices))

    # trainData(80,1)  ==> 80 data ptrs, with output of 1 Dimen
    # testData(10,1)   ==> 10 data ptrs, with output of 1 Dimen
    #print("traindata: ", trainData.shape)
    #print("testdata: ", testData.shape)

    result = []
    for s in tf.split(0, testData.shape[0], res.indices):
        v = np.zeros(trainData.shape[0])
        #print("s is: ", s, type(s), "\ncontent: ",sess.run(s))
        v[sess.run(s)] = 1/k
        result.append(v)

    tResult = tf.stack(result)
    #print(result)
    #print(tResult)
    #print(sess.run(tResult))
    return tResult
    #return result
    #tf.shape(result)

# [ [0 1]
#   [1 0]
#   [1 0] ]
    #print (testData.item(1, sess.run (res.indices[0])) )


def datagen():
    np.random.seed(521)
    Data = np.linspace(1.0, 10.0 , num =100) [:, np. newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
             + 0.5 * np.random.randn(100 , 1)
    # randIdx from 0,1,...99
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

    return (trainData, trainTarget,validData,
            validTarget, testData, testTarget
            ,Data, Target)


main()