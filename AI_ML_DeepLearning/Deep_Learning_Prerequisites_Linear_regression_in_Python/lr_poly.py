# lr_poly.py: python script for calculating linear regression of a polynomial

import matplotlib.pyplot as plt
import numpy as np

# load data
dfile = np.genfromtxt('data_poly.csv', delimiter=',')

#x = [dfile[i][0] for i in range(len(dfile))]
#X = [[1, x[i], x[i]*x[i]] for i in range(len(x))]
#X = np.array(X)
x = [[1, dfile[i][0], dfile[i][0]*dfile[i][0]] for i in range(len(dfile))]
y = [dfile[i][1] for i in range(len(dfile))]

# convert to arrays
x, y = np.array(x), np.array(y)

def plot_show_data():
    plt.clf()
    plt.scatter(x[:,1], y)
    plt.show()
#plot_show_data()

# multiple linear regression
w = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, y))
yhat = np.dot(x,w)

def plot_with_line():
    plt.clf()
    plt.scatter(x[:,1], y)
    #plt.plot(x[:,1], yhat)# comes out wonky, so you need to sort x and y
    plt.plot(sorted(x[:,1]), sorted(yhat), 'r')
    #plt.show()
    plt.savefig('figs/lr_poly.png')
plot_with_line()

d1 = y - yhat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print "R^2 value: ", r2#>> R^2 value:  0.999141229637



