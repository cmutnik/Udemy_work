import matplotlib.pyplot as plt
import numpy as np

# set number data points and dimensionality to 50
N, D = 50, 50
# x is uniform distribution of point centered around 0, from -5:5
X = (np.random.random((N, D)) - 0.5)*10
# everything else is 0, so last D-3 terms dont influence
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))
# set Y = X (dot) true_w + random gaussian noise
Y = X.dot(true_w) + np.random.randn(N)*0.5

# start gradient descent
costs = []
w = np.random.randn(D) / D**0.5
learning_rate = 0.001
l1 = 10.# l1 regularization rate
for i in xrange(500):
    Yhat = X.dot(w)# prediction
    delta = Yhat - Y
    # update w value with regularization term
    w = w - learning_rate*(X.T.dot(delta) + l1*np.sin(w))
    # find/store cost
    mse = delta.dot(delta) / N
    costs.append(mse)
print "final w: ", w

def plot1():
    plt.clf()
    plt.plot(true_w, label='true w')
    plt.plot(w, label='w map')
    plt.legend()
    plt.show()
    #plt.savefig('figs/l1_regularization.png')
plot1()

