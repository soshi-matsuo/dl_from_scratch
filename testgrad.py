import numpy as np
from dlbook.common.functions import softmax, cross_entropy_error
from dlbook.common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # mean=0, sigma=1の正規分布で初期化
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
net = simpleNet()
print(net.W)

x1 = np.array([0.6, 0.9])
p = net.predict(x1)
print(p, np.argmax(p))
t1 = np.array([1,0,0])

# 損失関数において、パラメータWの勾配dWを導出
fW = lambda W: net.loss(x1, t1)
dW = numerical_gradient(fW, net.W)
print(dW)