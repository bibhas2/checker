import numpy as np
import matplotlib.image as img

def load_image_samples():
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
