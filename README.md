# IMDB-Classifier
This is one of my very first encounters with Machine Learning. I specifically utilized the Keras kernel with a Tensorflow backdoor to provide the framework and tools to create and train my neural network. 

It parses an existing dataset of highly-polarized IMDB reviews (included in the Keras library), and trains a binary classification neural network to identify if a review is 'negative' or 'positive'. 

### Neural Network Model
I chose to utilize a sequential model for my neural network (NN), as I am dealing with a binary classification problem, thus I needed a model which can possess a linear stack of layers. 
My layers are as follows: 
1 - Dense layer (Activation function: 'relu')
2 - Dense layer (Activation function: 'relu')
3 - Dense layer (Activation function: 'sigmoid')

I utilized these the 'relu' function for my first two layers as I only needed to classify the types of IMDB reviews into the two categories mentioned above. The 'sigmoid' function is used in the final layer as I wish to return a scalar between 0 and 1 which encodes a probability that describes the 'confidence' of the predictions made by the NN on its training set.

### Loss Function
The loss function I used was 'binary_crossentropy' as one should use in any binary classification problem.

### Optimizer Choice
I chose to use the 'rmsprop' optimizer to train my network as it is general purpose and easy for beginners like me to understand.

To counteract overfitting of my network, I utilized 4 epochs to combat this effect. 

### Note
I used the pyplot library from matplotlib to graphically represent the loss and accuracy of my NN between training and testing phases.

### License
This project is authored under the [MIT](https://choosealicense.com/licenses/mit/) license.
