import matplotlib.pyplot as plt
import numpy as np

N, D = 4, 2

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
# target: False, True, True, False
T = np.array([0,1,1,0])

ones = np.array([[1]*N]).T

def plot_look_at_data():
    plt.clf()
    plt.scatter(x[:,0], x[:,1], c=T)
    plt.show()
#plot_look_at_data()
#>> Red_dot         Blue_dot
#>> Blue_dot        Red_dot

# change this 2D pronlem to a 3D problem, to make data linearly seperable
xy = np.matrix(x[:,0] * x[:,1]).T
xb = np.array(np.concatenate((ones, xy, x), axis=1))

##########
# Copy code from logistic2.py
##########
w = np.random.randn(D+2)
z = xb.dot(w)

def sigmoid(a):
    return 1/(1 + np.exp(-a))

y = sigmoid(z)

def cross_entropy_error(t, y):
    #J = -(t*np.log(y) + (1-t)*np.log(1-y))
    E=0
    for i in xrange(N):
        if t[i] == 1:
            E -= np.log(y[i])
        else:
            E -= np.log(1 - y[i])
    return E
##########
#print "Cross Entropy Error: ", cross_entropy_error(T,y)
#>> Cross Entropy Error:  4.5448936848

# let's do gradient descent 100 times
learning_rate = 0.01
error = []
for i in range(10000):
    e = cross_entropy_error(T, y)
    error.append(e)
    if i % 1000 == 0:
        print(e)

    # gradient descent weight udpate with regularization
    w += learning_rate * ( xb.T.dot(T - y) - 0.01*w )

    # recalculate Y
    y = sigmoid(xb.dot(w))

def plot_error_():
    plt.clf()
    plt.plot(error)
    plt.title("Cross-entropy per iteration")
    #plt.show()
    plt.savefig('figs/logistic_xor.png')
plot_error_()

print "Final w:", w
print "Final classification rate:", 1 - np.abs(T - np.round(y)).sum() / N
#>> Final w: [-1.48729704 -7.69123375  3.41571576  3.41571553]
#>> Final classification rate: 1.0
