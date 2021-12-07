from utils.DenseLayer import Dense


class Sequential():
    def __init__(self, Input_size):
        """
        Sequential model
        """
        self.layers = []
        self.params = []
        self.grads = []
        self.input_size = Input_size

    def add(self, layer_name, size_out):
        if self.layers == []:
            size_in = self.input_size
        else:
            size_in = self.layers[-1].size_out
        if layer_name == 'Dense':
            self.layers.append(Dense(size_in, size_out))
            # self.params.append([self.layers[-1].w, self.layers[-1].b])


    def loss(self, X, y):
        batch = X.shape[0]

        digits = X
        # forward
        for i in range(len(self.layers)):
            digits = self.layers[i].forward(digits)

        loss = (y-digits)**2/batch
        # backward
        # for i in range(len(self.layers))[::-1]:
        #     self.layers[i].backward()
        
        return loss