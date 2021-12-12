from utils.DenseLayer import Dense
from utils.Lossfunctions import softmax_loss
from utils.Classifiers import softmax
import numpy as np


LAYER_NAMES = ["Dense"]

class Sequential():
    def __init__(self, Input_size):
        """
        Sequential model
        """
        self.layers = dict()
        self.num_layers = 0
        self.name_layers = []
        # self.params = dict()
        # self.grads = dict()
        self.input_size = Input_size

    def add(self, layer_name, size_out):
        """
        Add layer to the sequential model
        
        Args:
        - layer_name: the type of layer to be appended (Dense, AveragePooling) 
        """
        if layer_name in LAYER_NAMES:
            self.num_layers += 1
        else:
            raise ValueError("Please use a correct layer name from: ", LAYER_NAMES)

        # pass input size of x to the added layer
        if self.layers:
            size_in = self.layers[self.name_layers[-1]].size_out
        else:
            size_in = self.input_size

        # give name to the added layer
        self.name_layers.append(layer_name + "_{}".format(self.num_layers))

        # add different types of layer
        if layer_name == 'Dense':
            self.layers[self.name_layers[-1]] = Dense(size_in, size_out)


    def loss(self, X, y):
        # batch = X.shape[0]

        digits = X
        # forward
        for key in self.name_layers:
            digits = self.layers[key].forward(digits)

        # y_onehot = np.eye(10)[y]
        # loss = (digits-y_onehot)**2/batch
        # dx = np.abs(digits-y_onehot)
        # print(loss)
        loss, dx = softmax_loss(digits, y)
        # print("loss:", loss)
        
        # print("dx",dx.shape)

        # backward
        for i in range(len(self.layers))[::-1]:
            # print(self.name_layers[i])
            dx = self.layers[self.name_layers[i]].backward(dx)
            
        
        return loss

    
    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        perdictions = None

        digits = X
        # forward
        for key in self.name_layers:
            digits = self.layers[key].forward(digits)

        # digits = softmax(digits)
        predictions = np.argmax(digits, axis=1)

        return predictions

    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))

        return acc



    def step(self, learning_rate=1e-3):
        """
        Use SGD to implement a single-step update to weight and bias.
        defaul learning rate is 0.001.

        Args:
        - learning_rate: default is 1e-3
        """
        # for each layer
        for key_layer in self.layers:
            # if this layer have trainable parameters
            if self.layers[key_layer].params:
                layer = self.layers[key_layer]
                # back propagation
                for k in layer.params:
                    layer.params[k] -= learning_rate * layer.grads[k]
