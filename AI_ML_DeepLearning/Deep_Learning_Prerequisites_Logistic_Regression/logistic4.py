# Python script that modifies logistic3.py to add regularization

##########
# Use code from logistic3.py (code commented in original file)
##########
import numpy as np

N, D = 100, 2
x = np.random.randn(N,D)

x[:50, :] = x[:50, :] - 2*np.ones((50, D))
x[50:, :] = x[50:, :] + 2*np.ones((50, D))

t = np.array([0]*50 + [1]*50)
ones = np.ones((N,1))

xb = np.concatenate((ones, x), axis=1)
w = np.random.randn(D+1)
z = xb.dot(w)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

y = sigmoid(z)

def cross_entropy_error(t, y):
    """J = -(t*np.log(y) + (1-t)*np.log(1-y))"""
    E=0
    for i in xrange(N):
        if t[i] == 1:
            E -= np.log(y[i])
        else:
            E -= np.log(1 - y[i])
    return E

print "Cross Entropy Error: ", cross_entropy_error(t,y)

learning_rate = 0.1
for i in xrange(100):
    if i % 10 == 0:
        print cross_entropy_error(t,y)
    # update weights, using regularization with \lambda=0.1
    w += learning_rate * (xb.T.dot(t-y) - 0.1*w)
    y = sigmoid(xb.dot(w))

# print out the final weight...regularization should give us smaller values for w
print "Final weight (regularization), w: ", w
#>> Final weight (regularization), w:  [ 3.68957294  6.0419555   5.87110711]
