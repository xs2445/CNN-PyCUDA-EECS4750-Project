from utils.Model import Sequential
import numpy as np
import matplotlib.pyplot as plt



input_size = 10
num_class = 4

model = Sequential(Input_size=input_size)
model.add('Dense', 50)
# model.add('Dense', 50)
model.add('Dense', num_class)
# model.add('Dense', o)


N = 2000

batch = 64

noise = np.random.rand(N, input_size)-0.5
y = np.random.randint(0,num_class,size=N)
X = np.empty_like(noise)

# y = y.astype(np.float32)
X = X.astype(np.float32)

for i in range(N):

    X[i] = (y[i]+noise[i])/10

from utils.Optimizers import SGDOptim, my_SGD

N_val = 0.1


X_train = X[:int(N*(1-N_val))]
y_train = y[:int(N*(1-N_val))]


X_val = X[int(N*(1-N_val)):]
y_val = y[int(N*(1-N_val)):]


optimizer = my_SGD()
lr = 1e-4
epoch = 20

loss_hist, train_acc_hist, valid_acc_hist = optimizer.train(model, X_train, y_train, X_val, y_val, 
        epoch, batch, lr, learning_decay=1)

plt.plot(loss_hist/np.max(loss_hist))
plt.plot(train_acc_hist)
plt.plot(valid_acc_hist)
plt.show()