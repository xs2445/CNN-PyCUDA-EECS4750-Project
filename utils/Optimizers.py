import numpy as np




class Optimizer(object):
    def train(self, model, X_train, y_train, X_valid, y_valid,
              num_epoch=10, batch_size=500, learning_rate=1e-3, learning_decay=0.95, verbose=False, record_interval=10):

        """
        This function is for training

        Inputs:
        :param model: (class MLP) a MLP model
        :param X_train: (float32) input data, a tensor with shape (N, D1, D2, ...)
        :param y_train: (int) label data for classification, a 1D array of length N
        :param X_valid: (float32) input data, a tensor with shape (num_valid, D1, D2, ...)
        :param y_valid: (int) label data for classification, a 1D array of length num_valid
        :param num_epoch: (int) the number of training epochs
        :param batch_size: (int) the size of a single batch for training
        :param learning_rate: (float)
        :param learning_decay: (float) reduce learning rate every epoch
        :param verbose: (boolean) whether report training process
        """
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        print('number of batches for training: {}'.format(num_batch))
        loss_hist = []
        train_acc_hist = []
        valid_acc_hist = []
        loss = 0.0
        for e in range(num_epoch):
            # Train stage
            for i in range(num_batch):
                # Order selection
                X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]
                # loss
                loss += model.loss(X_batch, y_batch)
                # update model
                self.step(model, learning_rate=learning_rate)

                if (i + 1) % record_interval == 0:
                    loss /= record_interval
                    loss_hist.append(loss)
                    if verbose:
                        print('{}/{} loss: {}'.format(batch_size * (i + 1), num_train, loss))
                    # loss = 0.0

            # Validation stage
            train_acc = model.check_accuracy(X_train, y_train)
            val_acc = model.check_accuracy(X_valid, y_valid)
            train_acc_hist.append(train_acc)
            valid_acc_hist.append(val_acc)
            # Shrink learning_rate
            learning_rate *= learning_decay
            print('epoch {}: loss = {}, train acc = {}, val acc = {}, lr = {}'.format(e + 1, loss, train_acc, val_acc, learning_rate))

        # Save loss and accuracy history
        self.loss_hist = loss_hist
        self.train_acc_hist = train_acc_hist
        self.valid_acc_hist = valid_acc_hist

        return loss_hist, train_acc_hist, valid_acc_hist

    def test(self, model, X_test, y_test, batch_size=10000):
        """
        Inputs:
        :param model: (class MLP) a MLP model
        :param X_test: (float) a tensor of shape (N, D1, D2, ...)
        :param y_test: (int) an array of length N
        :param batch_size: (int) seperate input data into several batches
        """
        acc = 0.0
        num_test = X_test.shape[0]

        if num_test <= batch_size:
            acc = model.check_accuracy(X_test, y_test)
            print('accuracy in a small test set: {}'.format(acc))
            return acc

        num_batch = num_test // batch_size
        for i in range(num_batch):
            X_batch = X_test[i * batch_size:(i + 1) * batch_size]
            y_batch = y_test[i * batch_size:(i + 1) * batch_size]
            acc += batch_size * model.check_accuracy(X_batch, y_batch)

        X_batch = X_test[num_batch * batch_size:]
        y_batch = y_test[num_batch * batch_size:]
        if X_batch.shape[0] > 0:
            acc += X_batch.shape[0] * model.check_accuracy(X_batch, y_batch)

        acc /= num_test
        print('test accuracy: {}'.format(acc))
        return acc

    def step(self, model, learning_rate):
        pass


class SGDOptim(Optimizer):
    def __init__(self):
        pass

    def step(self, model, learning_rate):
        """
        Implement a one-step SGD update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """

        # get all parameters and their gradients
        params = model.params
        grads = model.grads

        for k in grads:
            ## update each parameter
            params[k] -= learning_rate * grads[k]



class my_SGD(Optimizer):
    def __init__(self):
        pass

    def step(self, model, learning_rate=1e-3):
        """
        Use SGD to implement a single-step update to weight and bias.
        defaul learning rate is 0.001.

        Args:
        - learning_rate: default is 1e-3
        """
        # for each layer
        for key_layer in model.layers:
            # if this layer have trainable parameters
            # print(key_layer)
            if model.layers[key_layer].params:
                params = model.layers[key_layer].params
                grads = model.layers[key_layer].grads
                # back propagation
                for k in params:
                    # print(k)
                    params[k] -= learning_rate * grads[k]
                # model.layers[key_layer].params = params
                # model.layers[key_layer].grads = grads