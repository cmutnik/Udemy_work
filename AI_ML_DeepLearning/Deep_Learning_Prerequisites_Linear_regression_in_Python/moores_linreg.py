# linear regression of moors law
# data file from: https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/linear_regression_class/moore.csv

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# open data
#from astropy.io import ascii
#dfile = ascii.read('wantedmoore.txt')
#dfile

dfile = np.genfromtxt("wantedmoore.txt")

y_numval, year = [], []

for i in range(len(dfile)):
    y_numval.append(dfile[i][0])
    year.append(dfile[i][1])
#print year

def plot_raw():
    plt.clf()
    plt.plot(year,np.log(y_numval), 'ro', label='data')
    plt.xlabel('Year')
    plt.ylabel('log(N)')

def linfit(x, m, b):
    return m*x+b




######################################################################
#x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
#y = np.array([5, 20, 14, 32, 22, 38])
x=np.array(year).reshape((-1, 1))
y=np.log(y_numval)
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
#>> ('coefficient of determination:', 0.33375819946741436)
print('intercept:', model.intercept_)
#>> ('intercept:', -155669591648.4585)
print('slope:', model.coef_)
#>> ('slope:', array([ 78368485.06988795]))

######################################################################
new_model = LinearRegression().fit(x, np.array(y).reshape((-1, 1)))
print('intercept:', new_model.intercept_)
#>> ('intercept:', array([ -1.55669592e+11]))
print('slope:', new_model.coef_)
#>> ('slope:', array([[ 78368485.06988795]]))

def plot_models():
    plot_raw()
    y_fit = linfit(x, model.coef_, model.intercept_)

    plt.plot(x, y_fit, 'k--', label='model')
    plt.plot(x, linfit(x, new_model.coef_, new_model.intercept_), 'g-', label='new model')
    plt.legend(loc='topleft')

plot_models()