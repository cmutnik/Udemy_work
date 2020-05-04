import matplotlib.pyplot as plt
import numpy as np

###########
# copy code from logistic2.py, for data generation
###########
N, D = 100, 2
x = np.random.randn(N,D)
# set first gaussian cloud of 50 points centered at (x,y)=(-2,-2)
x[:50, :] = x[:50, :] - 2*np.ones((50, D))
# second at centered at (2,2)
x[50:, :] = x[50:, :] + 2*np.ones((50, D))
# create array of targets: first 50 set to 0, last 50 set to 1
t = np.array([0]*50 + [1]*50)
#ones = np.array([[1]*N]).T# "old" commented out in instructors code
ones = np.ones((N,1))
# concat them together
xb = np.concatenate((ones, x), axis=1)

"""
# This func is NOT used in this code, but the instructor included it
def sigmoid(a):
    return 1/(1 + np.exp(-a))
"""
# closed form solution
w = np.array([0, 4, 4])

# this should all give us a line (y = -x); so plot it to see
def plot_pts_line():
    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], s=100, c=t, alpha=0.5)
    # set x-axis [-6, 6] with 100 points inbetween
    x_ax = np.linspace(-6, 6, 100)
    y_ax = -x_ax
    # plot the line
    plt.plot(x_ax, y_ax)
    #plt.show()
    # save fig instead of showing it
    plt.title('Made From logistic_visualize.py')
    plt.savefig('figs/logistic_visualize.png')

plot_pts_line()
