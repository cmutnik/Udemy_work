# Practical problem in logistical regression - the donut problem
import matplotlib.pyplot as plt
import numpy as np

# more data points to see things more easily
N, D = 1000, 2

R_inner, R_outer = 5, 10
# set uniformly distributed variable for half the data that depends on inner raius (its spread around 5)
R1 = np.random.randn(N/2) + R_inner
theta = 2*np.pi*np.random.random(N/2)
# convert polar coords to x,y...transpose, so N goes along the rows
x_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
# do the same for the outter radius
R2 = np.random.randn(N/2) + R_outer
theta = 2*np.pi*np.random.random(N/2)
x_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

# calculate entire x
x = np.concatenate([ x_inner, x_outer ])
# first set of targets is 0, second set is 1
T = np.array([0]*(N/2) + [1]*(N/2))

def plot_look_at_data():
    """plot to see what is looks like"""
    plt.clf()
    plt.scatter(x[:,0], x[:,1], c=T)
    #plt.show()
    plt.savefig('figs/donut_problem')
plot_look_at_data()

# create column of ones for bias term
ones = np.array([[1]*N]).T
# for donut problem we need another column for the radius of the point
r = np.zeros((N,1))
for i in xrange(N):
    r[i] = np.sqrt(np.dot(x[i,:], x[i,:]))

xb = np.concatenate((ones, r, x), axis=1)
# randomly initialize some weights
w = np.random.randn(D+2)

z = xb.dot(w)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

y = sigmoid(z)

def cross_entropy(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

learning_rate = 0.0001
error = []
for i in xrange(5000):
    e = cross_entropy(T, y)
    error.append(e)
    # print it out every 100 times
    if i % 100 == 0:
        print e
    # use gradient descent with regularization
    w += learning_rate * (np.dot((T-y).T, xb) - 0.01*w)
    # recalculate the output
    y = sigmoid(xb.dot(w))

def plot_error_over_time():
    plt.clf()
    plt.plot(error)
    plt.title("Cross-enetropy")
    #plt.show()
    plt.savefig('figs/donut_error.png')
plot_error_over_time()
print "Final w: ", w
print "Final classification rate: ", 1-np.abs(T-np.round(y)).sum() / N
#>> Final w:  [ -1.16486685e+01   1.58642040e+00  -5.41041735e-03   9.48361149e-03]
#>> Final classification rate:  0.995
