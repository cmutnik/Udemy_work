import matplotlib.pyplot as plt
import numpy as np

# Number of data points evenly spaces between 0-10
N=50
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)# half x + random noise

# make outliers
Y[-1] += 30
Y[-2] += 30

def plot_lookAtData():
    plt.clf()
    plt.scatter(X,Y)
    plt.show()
#plot_lookAtData()

# add bias term
X = np.vstack([np.ones(N), X]).T

# calculate maximum likelihood solution
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# predictions for w_ml
Yhat_ml = X.dot(w_ml)

def plot_w_line():
    """plot original data and maximum likelihood line"""
    plt.clf()
    plt.scatter(X[:,1], Y)
    plt.plot(X[:,1], Yhat_ml, 'r-')
    plt.show()
#plot_w_line()

# l2 regularization solution
l2 = 1000.# set l2 penalty to 1000
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))# w2 solution derived in video
#   note: np.eye(i) makes an identity matrix of dimensions ixi
Yhat_map = X.dot(w_map)
# inspect plot
def plot_l2reg_line():
    """plot original data, maximum likelihood line, and l2 reg line"""
    plt.clf()
    plt.scatter(X[:,1], Y)
    plt.plot(X[:,1], Yhat_ml, 'r-', label='max likelihood')
    plt.plot(X[:,1], Yhat_map, 'k-', label='map')
    plt.legend()
    plt.show()
    #plt.savefig('figs/l2_regularization.png')
plot_l2reg_line()

