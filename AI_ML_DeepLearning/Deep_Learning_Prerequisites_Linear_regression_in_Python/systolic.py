# Udemy linear regression course

# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import matplotlib.pyplot as plt
import numpy as np

dfile = np.genfromtxt('systolic.csv', delimiter=',')

x1 = [dfile[i][0] for i in range(len(dfile))]
x2 = [dfile[i][1] for i in range(len(dfile))]
x3 = [dfile[i][2] for i in range(len(dfile))]

def plot_data():
    plt.clf()
    plt.xlabel('Age')
    plt.plot(x2, x1, 'ro', label="Systolic BP")
    plt.plot(x2, x3, 'bo', label="Weight (lb)")
    plt.legend(loc="topleft")
    plt.savefig("figs/systolic_data.png")
#plot_data()


Y = x1
X, X2only, X3only = [], [], []
for j in range(len(x2)):
    X.append([x2[j], x3[j], 1])
    X2only.append([x2[j], 1])
    X3only.append([x3[j], 1])

Y, X, X2only, X3only = np.array(Y), np.array(X), np.array(X2only), np.array(X3only)

def get_r2(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)

    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2

print "r2 for x2 only: ", get_r2(X2only, Y)
print "r2 for x3 only: ", get_r2(X3only, Y)
print "r2 for both: ", get_r2(X, Y)
#>> r2 for x2 only:  0.957840720815
#>> r2 for x3 only:  0.941995208529
#>> r2 for both:  0.97684710415

