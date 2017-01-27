import numpy as np
import matplotlib.image as img
import tensorflow as tf

def load_training_samples():
    NUM_SAMPLES = 10
    TRAINING_DIR = "training_images"
    imageResult = []
    imageClassList = ["o", "c"]
    classResult = []

    for imageClassIndex in range(len(imageClassList)):
        for i in range(0, NUM_SAMPLES):
            #Load the image. The files don't have alpha channel.
            #image will be a 20x20x3 matrix
            imageFile = "{0}/{1}{2}.png".format(TRAINING_DIR, imageClassList[imageClassIndex], i + 1)
            print("Loading:", imageFile)
            image = img.imread(imageFile)

            #Average the RGB pixel values and create
            #a greyscale image matrix. grey_image will be a 20x20 matrix
            grey_image = image.mean(2)

            #Convert the 2D array into 1D
            grey_image = np.reshape(grey_image, grey_image.shape[0] * grey_image.shape[1])

            trainingClassification = np.zeros(len(imageClassList))
            trainingClassification[imageClassIndex] = 1

            #Append this image to the result
            imageResult.append(grey_image)
            classResult.append(trainingClassification)

    return (np.asarray(imageResult), np.asarray(classResult))

def runModel(model):
    sess = tf.Session()

    res = sess.run(model, {X: xArr, Y: yArr})

    print res

tf.set_random_seed(0)

samples = load_training_samples()
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

#print samples[1].shape

X = tf.placeholder(tf.float32, [4, 2])
#X = tf.placeholder(tf.float32, [4])
Y = tf.placeholder(tf.float32, [2, 4])

# xArr = np.array([
#     [1, 2],
#     [3, 4],
#     [5, 6],
#     [7, 8]
# ])
# xArr = np.array([
#     1, 2, 3, 4
# ])
# yArr = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8]
# ])

# model = tf.reduce_sum(X * tf.log(X))
# runModel(model)

# model = tf.nn.softmax(X)
# runModel(model)

# model = tf.matmul(X, Y)
# runModel(model)

# model = tf.reshape(X, [1, 8])
# runModel(model)

# model = tf.add(X, X)
# runModel(model)


