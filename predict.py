import numpy as np
import tensorflow as tf
from image_loader import load_image_samples

tf.set_random_seed(0)

samples = load_image_samples()
testImages = samples[0]
trainClassification = samples[1]
#Number of classes
K = trainClassification[0].shape[0]
#Number of training samples
m = testImages.shape[0]
#Number of features
n = testImages.shape[1]

print("Number of classes:", K)
print("Number of test samples:", m)
print("Number of features:", n)

#Load the trained weights and biases.
trainWeights = np.load("weights.npy")
trainBiases = np.load("biases.npy")

#Training data placeholder
X = tf.placeholder(tf.float32, [m, n])
#Training prediction placeholder
Y_ = tf.placeholder(tf.float32, [m, K])
#Weights
W = tf.placeholder(tf.float32, [n, K])
#Biases
b = tf.placeholder(tf.float32, [K])

# The operation that calculates predictions
Y = tf.nn.softmax(tf.matmul(X, W) + b)

#Cost function
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_data={X: testImages, Y_: trainClassification, W: trainWeights, b: trainBiases}

#Apply the final weights and biases on the training data
checkResult = sess.run(Y, feed_dict=train_data)

print(checkResult)
