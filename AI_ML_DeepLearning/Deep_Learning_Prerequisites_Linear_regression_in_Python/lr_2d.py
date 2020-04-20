# lr_2d.py: multidimensional linear regression 
# dat file data_d2.csv: col1-3 respectively are x1, x2, y
# x have 2 dimensions, y has 1
# D=2; N=100

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# open data file
dfile = np.genfromtxt('data_2d.csv', delimiter=',')

"""
x1 = [dfile[i][0] for i in range(len(dfile))]
x2 = [dfile[i][1] for i in range(len(dfile))]
y = [dfile[i][2] for i in range(len(dfile))]
# make list of x values, saved as 2d arrays
X = []
for i in range(len(dfile)):
    X.append([float(dfile[i][0]), float(dfile[i][1]), 1])
"""

# done more cleanly with list comprehension
X = [[float(dfile[i][0]), float(dfile[i][1]), 1] for i in range(len(dfile))]
Y = [dfile[i][2] for i in range(len(dfile))]

# convert to arrays
X, Y = np.array(X), np.array(Y)

def plot_3d_scatter():
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y)
    plt.show()
#plot_3d_scatter()

# calculate weights of our model
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# calculate R^2
d1 = Y - Yhat
d2 = Y - np.mean(Y)
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print "R^2 value: ", r2
#>> R^2 value:  0.998004061248

def plot_3d_scatter_line():
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y)
    ax.plot(sorted(X[:,0]), sorted(X[:,1]), sorted(Yhat))
    plt.savefig('figs/lr_2d.png')

# uncomment to have plot made
#plot_3d_scatter_line()

