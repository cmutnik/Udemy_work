# python script to...
from process import get_binary_data
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

# get binary data and shuffle it (so its not in order)
x, y = get_binary_data()
x, y = shuffle(x, y)

# split into train and test sets
xtrain, ytrain = x[:-100], y[:-100]
xtest, ytest = x[-100:], y[-100:]

# randomly initalize weights again
D = x.shape[1]
w = np.random.randn(D)
b = 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward(X, W, b):
    return sigmoid(np.dot(X,W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

# takes in targets and P of y given x: P(y|x)
def cross_entropy(T, pY):
    return -np.mean( T*np.log(pY) + (1-T)*np.log(1-pY) )

##########
# Enter main training loop
##########
# keep arrays of training and test costs
train_costs, test_costs = [], []
learning_rate = 0.001
for i in xrange(10000):
    # calculate values
    pYtrain = forward(xtrain, w, b)
    pYtest = forward(xtest, w, b)

    # calculate training/test costs and append to lists of costs
    #train_costs.append(cross_entropy(ytrain, pYtrain))# dont consolidate, values are printed below
    #test_costs.append(cross_entropy(ytest, pYtest))# dont consolidate, values are printed below
    ctrain = cross_entropy(ytrain, pYtrain)
    ctest = cross_entropy(ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    ##########
    # Now we can start gradient descent
    ##########
    # use vectorized version of these equations
    w -= learning_rate * xtrain.T.dot(pYtrain - ytrain)
    b -= learning_rate * (pYtrain - ytrain).sum()
    # print training/test costs every 1000 steps
    if i % 1000 == 0:
        print i, ctrain, ctest

print "Final train classification rate: ", classification_rate(ytrain, np.round(pYtrain))
print "Final test classification rate: ", classification_rate(ytest, np.round(pYtest))
#>> Final train classification rate:  0.973154362416
#>> Final test classification rate:  0.95

def plot_costs():
    plt.style.use("ggplot")
    plt.clf()
    plt.plot(train_costs, label='train cost')
    plt.plot(test_costs, label='test cost')
    plt.legend()
    plt.title('Made Using logistic_train.py')
    plt.savefig('figs/logistic_train.png')
plot_costs()
