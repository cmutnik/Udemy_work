# Python script to demonstrate use of l1 regularization (l1 regularization helps us achieve sparsity)
#   generate data that has a large matrix as its input
#   y only depends on a few features from the data, the rest is just noise
#   use l1 regularization to find a sparse set of weights that id the useful dimensions of x
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

N, D = 50, 50

# set x to be uniformly distributed numbers between -5 and 5, subtract 0.5 to center around 0
x = (np.random.randn(N,D) - 0.5)*10

# have only the first 3 dimensions effect the output and the rest are 0
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

# generate the targets, y, with some added random noise
y = np.round(sigmoid(x.dot(true_w) + np.random.randn(N)*0.5))

# perform gradient descent to find w
# keep track of squared error cost
costs = []
# randomly initialize weights
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
# set l1 penalty...see what happens when you change this to different values
l1 = 2.0
#l1=10.

for i in xrange(5000):
    yhat = sigmoid(x.dot(w))
    delta = yhat - y
    # gradient descent
    w = w - learning_rate*(x.T.dot(delta) + l1*np.sign(w))
    # find and store the costs
    cost = -(y*np.log(yhat) + (1-y)*np.log(1 - yhat)).mean() + l1*np.abs(w).mean()
    costs.append(cost)

print "Final w: ", w
#>> Final w:  [  4.80188713e-01   1.68028068e-01  -2.84033176e-01   6.37687263e-04

def plot_costs():
    plt.clf()
    plt.plot(costs)
    plt.title('Costs (From l1_regularization.py)')
    plt.ylim(0,np.max(costs)*0.1)
    #plt.show()
    plt.savefig('figs/l1_regularization_costs_l1penatly_'+str(l1)+'.png')
plot_costs()

def plot_W_vs_trueW():
    plt.clf()
    plt.plot(true_w, label='true w')
    plt.plot(w, label='w map')
    plt.legend()
    #plt.title('Compare w (From l1_regularization.py) with l1 penalty of '+str(l1))
    plt.title('Compare w (From l1_regularization.py)')
    #plt.show()
    plt.savefig('figs/l1_regularization_compareW_l1penalty_'+str(l1)+'.png')
plot_W_vs_trueW()


































