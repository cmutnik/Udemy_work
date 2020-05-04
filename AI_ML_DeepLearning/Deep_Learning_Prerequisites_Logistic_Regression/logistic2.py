# Python script to calculate cross entropy error
import numpy as np

N, D = 100, 2
x = np.random.randn(N,D)

# set first 50 points centered at (x,y)=(-2,-2)
x[:50, :] = x[:50, :] - 2*np.ones((50, D))
# centered at (2,2)
x[50:, :] = x[50:, :] + 2*np.ones((50, D))
# create array of targets: first 50 set to 0, last 50 set to 1
t = np.array([0]*50 + [1]*50)
# need it to be 2D, to have N rows and 1 column
#ones = np.array([[1]*N]).T# "old" commented out in instructors code
ones = np.ones((N,1))
# concat them together
xb = np.concatenate((ones, x), axis=1)

# initialize random weights
w = np.random.randn(D+1)
# model output: z
z = xb.dot(w)

def sigmoid(a):
    return 1/(1 + np.exp(-a))

y = sigmoid(z)

# calculate the cross extropy error (depends on targets and predicted output)
def cross_entropy_error(t, y):
    #J = -(t*np.log(y) + (1-t)*np.log(1-y))
    E=0
    for i in xrange(N):
        if t[i] == 1:
            E -= np.log(y[i])
        else:
            E -= np.log(1 - y[i])
    return E
#print N == len(t)#>> True

print "Cross Entropy Error: ", cross_entropy_error(t,y)
#>> Cross Entropy Error:  69.6206716259


############
# check how good the closed form solution to logistic regression is
############
"""
# Closed form solution J
J = -(t*np.log(y) + (1-t)*np.log(1-y))
#print J
yh=sigmoid(J)
print cross_entropy_error(t,yh)#>> 9.43346309873
"""
# equal variances, so weights only depend on means
w = np.array([0, 4, 4])# bias=0, weights are 4

z = xb.dot(w)
y = sigmoid(z)

print "Cross Entropy Error (closed form): ", cross_entropy_error(t,y)
#>> Cross Entropy Error (closed form):  0.0708535952774
