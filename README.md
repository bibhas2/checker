##The problem
Most beginner level neural network projects work with predefined data sets like MNIST.
But I thought it will be appropriate to create my own problem domain and own dataset.
The problem we are trying to solve here is recognition of hand drawn symbols. Eventually
there will be three symbols:

- Circle
- Tick mark
- Plus sign

##Preparing the data
Each sample image is 20x20 PNG file. I had to strip out the alpha channel from each file.
Otherwise the RGB values were strange for some reason. When loading a file I do an average
of the RGB values and squish three values into a single greyscale value. So basically,
each image has 20x20 or 400 features.

##The model
While watching Martin Görner's excellent talk **Learn TensorFlow and deep learning, without a Ph.D.**
I learned that it is possible to have a single layer neural network. Why didn't I think about this before.
A single layer NN is darn easy to understand and code. And it produces unbelivable accuracy.

##Lessons learned
This is my very first Tensorflow project. I did everything from scratch, define the problem,
prepare the training and test data and finally write the code. I had taken the code from
Martin Görner's talk and changed the shapes of the placehodlers and variables 
not to be hard coded. 