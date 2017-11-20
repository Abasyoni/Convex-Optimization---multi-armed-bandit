import numpy as np
import pandas as pd
import time
import skimage.measure
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


EPSILON = 0.00001

class Results(object):
    def __init__(self, objective, time):
        self.objective = objective
        self.time = time
        self.iter = len(objective)


## this function runs coordinate, gradient, and multi-armed bandit on coordinate, gradient, 
## and plots the results to compare perfomances
def main():
    # Preprocess the data - 
    #i) Pooling/Parsing 
    #ii)PCA to reduce dimensions
    (data, label) = processData()
    Length = (label.shape[0]) # 8816
    Features = (data.shape[1]) ## 785


    #initialize weights
    init_w = np.random.rand(1,Features)*1e-3


    # Call individual descent algo - make a note of the obj func, accuracy, time taken
    grad_results = grad_desc(init_w, data, label)
    #coord_results = coord_desc(init_w, data, label)
    #plt.plot(grad_results.objective)
    print("hi")
    print(grad_results.objective)
    #plt.show()



    # Implement UCB2 for MAB 


    # Report Algo used/epoch, net time taken, obj value achieved, net accuracy

def processData():
    # first, parse the data
    # /home/oyku/Desktop/Oyku/Convex 10-725/
    mnist = pd.read_csv('train_mnist.csv')

    mnist = pd.DataFrame.as_matrix(mnist)

    mnist_shuffled = shuffle(mnist)

    label = mnist[:,0]
    data = mnist[:,1::]

    #To use only 0s and 1s - reduced data

    data = data[(label < 2), :] # 8816
    label = label[(label < 2)] # 8816
    
    #If the above is removed - enter dim reduction here

    label = label.reshape([label.shape[0], 1])

    mean = np.mean(data)
    data = data - mean
    
    data = np.append(data, np.ones([data.shape[0],1]), axis = 1)
    
    return (data, label)

def calc_objective(w, data, label):
    result = (label.T - w.dot(data.T))
    result = 0.5*result.dot(result)
    return result

def calc_grad(w, data, label):
    return -(label.T - w.dot(data.T)).dot(data)

def grad_step(w, data, label):
    step = 0.000001
    return w - step*calc_grad(w, data, label)

def grad_desc(W, data, label):
    w = W[:,:]

    objectives = []

    cur_time = time.time()
    objective = calc_objective(w, data, label)
    objectives += [objective]
    while (True):
        w = grad_step(w, data, label)
        new_objective = calc_objective(w, data, label)
        print new_objective - objective
        if (objective-new_objective < EPSILON):
            print("brekaing quickly")
            break
        objective = new_objective
        print("one more")
        objectives += [objective]
    cur_time = time.time() - cur_time

    return Results(objectives, cur_time)


main()

