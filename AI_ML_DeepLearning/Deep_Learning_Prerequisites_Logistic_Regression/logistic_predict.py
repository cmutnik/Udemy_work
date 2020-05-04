import numpy as np
from process import get_binary_data

# get data, set D to the dimensionality of data 
x, y = get_binary_data()
D = x.shape[1]
# initialize weights of logisitic regression model, w; bias term b
w = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1/(1+np.exp(-a))

def forward(x, w, b):
    return sigmoid(x.dot(w) + b)

# call forward function: wx + b
p_y_given_x = forward(x, w, b)
# make predictions
predictions = np.round(p_y_given_x)

def classification_rate(y, p):
    """ takes in targets, y, and predictions, p...returns mean (divides number correct by total number) """
    return np.mean(y==p)

print "Score [% accuracy]: ", classification_rate(y, predictions)
#>> Score [% accuracy]: 0.610552763819
