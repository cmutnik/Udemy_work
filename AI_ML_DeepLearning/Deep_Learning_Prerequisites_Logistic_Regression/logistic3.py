# Python script for gradient descent, with updating weights
import numpy as np
##########
# copy old code from logistic2.py
##########
N, D = 100, 2
x = np.random.randn(N,D)

# two normally distributed classes, each made of 50 points
#   first centered at (x,y)=(-2,-2)
x[:50, :] = x[:50, :] - 2*np.ones((50, D))
#   second at centered at (2,2)
x[50:, :] = x[50:, :] + 2*np.ones((50, D))

# create array of targets: first 50 set to 0, last 50 set to 1
t = np.array([0]*50 + [1]*50)
#ones = np.array([[1]*N]).T# "old" commented out in instructors code
ones = np.ones((N,1))
# concat them together
xb = np.concatenate((ones, x), axis=1)

##########
# randomly initialize weights
w = np.random.randn(D+1)
# model output calculated as
z = xb.dot(w)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

y = sigmoid(z)

# calculate the cross extropy error (depends on targets and predicted output)
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
#>> Cross Entropy Error:  42.4071030748

# use learning rate of 0.1; on 100 iterations of gradient descent
learning_rate = 0.1
for i in xrange(100):
    # print out cross entropy error every 10 steps, to check it is decreasing
    if i % 10 == 0:
        print cross_entropy_error(t,y)
    # update weights
    w += learning_rate * xb.T.dot(t-y)# learning_rate * np.dot((t-y).T, xb)
    # recalculate output, y
    y = sigmoid(xb.dot(w))

# print oout the final weight
print "Final weight, w: ", w
#>> Final weight, w:  [ -1.50911453  10.1079768   10.86793105]
