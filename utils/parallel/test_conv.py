import numpy as np
from ConvLayerPara import ConvLayerP

layer = ConvLayerP()

batch = 100
C, M, H, W, K = 3, 16, 60, 60, 3

x_shape = (batch, C,H,W)
m_shape = (M,C,K,K)
y_shape = (batch, M,H-K+1,W-K+1)

X = np.random.rand(*x_shape).astype(np.float32)
print('Shape of X: ', X.shape)
Masks = np.random.rand(*m_shape).astype(np.float32)
# print(Masks)
print('Shape of Masks: ', Masks.shape)

Y_shared = layer.forward_shared(X, Masks, batch, C, M, H, W, K)
# print(Y_shared)
