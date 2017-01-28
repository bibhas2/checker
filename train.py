import numpy as np
import matplotlib.image as img
import tensorflow as tf
from image_loader import load_image_samples

tf.set_random_seed(0)

samples = load_image_samples()
trainImages = samples[0]
trainClassification = samples[1]
#Number of classes
K = trainClassification[0].shape[0]
#Number of training samples
m = trainImages.shape[0]
#Number of features
n = trainImages.shape[1]

print("Number of classes:", K)
print("Number of training samples:", m)
print("Number of features:", n)

#Training data placeholder
X = tf.placeholder(tf.float32, [m, n])
#Training prediction placeholder
Y_ = tf.placeholder(tf.float32, [m, K])
#Weights
W = tf.Variable(tf.zeros([n, K]))
#Biases
b = tf.Variable(tf.zeros([K]))

# The operation that calculates predictions
Y = tf.nn.softmax(tf.matmul(X, W) + b)

#Cost function
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_graph = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

train_data={X: trainImages, Y_: trainClassification}

for i in range(1000):
    # train
    sess.run(train_graph, feed_dict=train_data)

    #You can calculate the cost after each iteration if you want.
    #It should steadily decline
    # cost = sess.run(cross_entropy, feed_dict=train_data)
    # print("Cost:", cost)

#Apply the final weights and biases on the training data
checkResult = sess.run(Y, feed_dict=train_data)

print(checkResult)
