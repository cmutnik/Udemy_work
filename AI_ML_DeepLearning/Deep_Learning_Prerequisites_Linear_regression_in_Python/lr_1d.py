# Linear Regression in 1D
import numpy as np
import matplotlib.pyplot as plt

"""
# script to generate 1D date: generate_1d.py
# instead of running this code, just get data from github file: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/linear_regression_class/data_1d.csv

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np

N = 100
with open('data_1d.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=N)
    Y = 2*X + 1 + np.random.normal(scale=5, size=N)
    for i in range(N):
        f.write("%s,%s\n" % (X[i], Y[i]))
"""


# load data
dfile = np.genfromtxt('data_1d.csv', delimiter=',')
"""
x_old,y_old = [], []
for i in range(len(dfile)):
    x_old.append(dfile[i][0])
    y_old.append(dfile[i][1])
print x_old
"""
# load data using list comprehension
x = [dfile[i][0] for i in range(len(dfile))]
y = [dfile[i][1] for i in range(len(dfile))]
#x == x_old#>> True
#y == y_old#>> True

# convert to arrays
x, y = np.array(x), np.array(y)

def ploty1():
    """ Plot the data to see what it looks like """
    plt.clf()
    plt.plot(x, y, 'ko')
    plt.xlabel('X AXIS')
    plt.ylabel('Y AXIS')
    plt.show()# needed if you run 'ipython' not if you run interative 'ipy'
# run plot function
#ploty1()

#####
# Apply equations from lecture
#####
# common denominator: N\Sigma(x_{i}^2) - \Sigma(x_i)^2
# note: x.dot(x) == np.dot(x,x)
#denom = len(x)*np.dot(x,x) - np.sum(x)**2

# how he does this in the video
denom2 = x.dot(x) - x.mean() * x.sum()#>> 71642.992720996961 ~= denom/100 = denom/len(x)

# calculate numerators for a and b
a = np.dot(y,x) - ( x.sum() * y.mean() )
b = ( np.mean(y) * x.dot(x) ) - ( np.mean(x) * np.dot(y,x) )
# divide by denominator
a = a/denom2
b = b/denom2

# calculate predicted y
y_hat = a*x + b

def plot_w_line():
    plt.clf()
    plt.scatter(x,y)
    plt.plot(x, y_hat, 'r-')
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('1D Linear Regression')
    #plt.show()
    plt.savefig('lr_1d.png')

# UNCOMMENT TO GET PLOT
#plot_w_line()

####################################################################################
# Above follows the instructors code, below uses the original equations
####################################################################################

denom_orig = len(x)*np.dot(x,x) - np.sum(x)**2
a_orig = ( len(x) * y.dot(x) ) - ( x.sum() * y.sum() ) 
b_orig = ( y.sum() * x.dot(x) ) - ( x.sum() * y.dot(x))

a_orig = a_orig / denom_orig
b_orig = b_orig / denom_orig


y_hat_orig = a_orig*x + b_orig


def plot_w_line_orig():
    # plot original and reduced equations, for comparison
    plt.clf()
    plt.scatter(x,y)
    plt.plot(x, y_hat, 'r-', lw=5, label='Reduced Equations')
    plt.plot(x, y_hat_orig, 'b--', label='Original Equations')
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('1D Linear Regression')
    plt.legend()
    #plt.show()
    plt.savefig('lr_1d_orig.png')
#plot_w_line_orig()


####################################################################################
# R^2 calculations
####################################################################################

# Try to calculate the R-squared value or "goodness of fit"
# Eqn: R^2 = 1 - ( SS_res / SS_tot )
#   SS: sum of square; res: residual; tot: total
# SS_res is the squared error from before: \Sigma (y_i - \hat{y_i})^2
# SS_tot: \Sigma (y_i - \bar{y})^2
#
# Eqn: R^2 = 1 - ( SS_res / SS_tot ) = \Sigma (y_i - \hat{y_i})^2 / \Sigma (y_i - \bar{y})^2
# Notes:
# Perfect Fit: R^2 = 1
# Overfit: R^2 = 0; we predicted the average of y
# Model worse than just using mean: R^2 < 0; SS_res/SS_tot > 1; using mean is better than this model

# calculate R^2 value
r2_num = np.sum( (y - y_hat)**2 )
r2_den = np.sum( (y - y.mean())**2 )

r2 = 1 - (r2_num / r2_den)
print 'R^2 value: ', r2
#>> R^2 value:  0.991183820298
"""
# calculate R^2 value using instructors method
his_r2_num = y - y_hat
his_r2_den = y - y.mean()
# treat them like a dot product, it multiplies the same element and sums
his_r2 = 1 - his_r2_num.dot(his_r2_num) / his_r2_den.dot(his_r2_den)#>> 0.99118382029778052

r2 == his_r2#>> True, so both methods work the same
"""

# calculate for un-reduced equations
r2_num_orig = np.sum( (y - y_hat_orig)**2 )
r2_den_orig = np.sum( (y - y.mean())**2 )

r2_orig = 1 - (r2_num_orig / r2_den_orig)
print 'R^2 value of original equations: ', r2_orig



#####
# PRINT R2 VALUE ON PLOT
#####
def plot_w_fit_r2():
    # plot original and reduced equations, for comparison
    plt.clf()
    plt.scatter(x,y)
    plt.plot(x, y_hat, 'r-', lw=5, label='Reduced Equations')
    plt.plot(x, y_hat_orig, 'b--', label='Original Equations')
    plt.xlabel('X Data')
    plt.ylabel('Y Data')
    plt.title('1D Linear Regression')
    text_plot = "R^2 value of both fits: " + str(r2)
    plt.text(-10, 200, text_plot)# print the R2 value on the graph, R2 for both fits is the same
    plt.legend()
    plt.savefig('figs/lr_1d_orig.png')
plot_w_fit_r2()
