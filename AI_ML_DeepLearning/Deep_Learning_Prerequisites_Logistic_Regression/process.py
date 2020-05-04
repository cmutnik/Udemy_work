import pandas as pd
import numpy as np

def get_data():
    # read data
    df = pd.read_csv('ecommerce_data.csv')
    # turn data into numpy matrix
    data = df.as_matrix()
    # split into x and y: y is last column, x is everything else
    x = data[:, :-1]
    y = data[:, -1]
    # normalize numerical columns, where normalized_data = (data - mean) / std_dev
    x[:,1] = (x[:,1] - np.mean(x[:,1]))/np.std(x[:,1])
    x[:,2] = (x[:,2] - np.mean(x[:,2]))/np.std(x[:,2])
    
    #  work on categorical column: time of day
    N, D = x.shape# get shape of original x
    x2 = np.zeros((N, D+3))# make new x, with sape: Nx(D+3)
    x2[:, 0:(D-1)] = x[:, 0:(D-1)]

    # one hot encoding for other 4 cols: loop thr
    for i in range(N):
        t = int(x[i,D-1])# get time of day, possible values: 0, 1, 2, 3
        x2[i, t+D-1] = 1# set values in x2
    # instead of loop: make new matrix of size Nx4 and then index Z directly
    Z = np.zeros((N,4))
    Z[np.arange(N), x[:,D-1].astype(np.int32)] = 1
    # UNCOMMENT IF DOING IT THIS WAY (NO LOOP)
    #x2[:,-4:] = z
    smallValue = 10e-10
    # to test it
    assert(np.abs(x2[:,-4:] -Z).sum() < smallValue)

    return x2, y

def get_binary_data():
    """ Here we only want binary data, not full data set (in logistics class) """
    x, y = get_data()
    # only take classes 0 and 1
    x2 = x[y <= 1]
    y2 = y[y <= 1]

    return x2, y2
