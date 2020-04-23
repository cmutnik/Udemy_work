import matplotlib.pyplot as plt
import numpy as np

# number data point is 10, dimensionality is 3 
N = 10 
D = 3
# set x to be matix of dim: NxD
X = np.zeros((N, D))
# set bias term
X[:,0] = 1
# set first 5 elements of 1st col, and last 5 elements of 2nd col to one
X[:5,1] = 1
X[:5,2] = 1
# y is 0 for first half of data, 1 for second half
Y = np.array([0]*5 + [1]*5)
# see what X looks like
print "X: ", X
#>> X:  [[ 1.  1.  1.]
#>>  [ 1.  1.  1.]
#>>  [ 1.  1.  1.]
#>>  [ 1.  1.  1.]
#>>  [ 1.  1.  1.]
#>>  [ 1.  0.  0.]
#>>  [ 1.  0.  0.]
#>>  [ 1.  0.  0.]
#>>  [ 1.  0.  0.]
#>>  [ 1.  0.  0.]]

# implement gradient descent
costs = []
# initialize w
w = np.random.randn(D) / D**0.5
# set learning rate
learning_rate = 0.001
for t in xrange(1000):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * X.T.dot(delta)# update w
    # find costs vis mean square error (mse)
    mse = delta.dot(delta) / N
    costs.append(mse)

def plot_cost():
    """ Visualize data """
    plt.clf()
    plt.plot(costs)
    plt.show()
#plot_cost()

# check value of w
print "final w: ", w
#>> final w:  [ 0.97956129 -0.23806796 -0.73302735]

def plot_y_yhat():
    """ confirm solution """
    plt.clf()
    plt.plot(Y, label='target')
    plt.plot(Yhat, label='predictions')
    plt.legend(loc='topleft')
    #plt.show()
    plt.savefig('figs/gradient_descent_solution.png')
plot_y_yhat()