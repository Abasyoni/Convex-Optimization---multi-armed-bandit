import numpy as np
import pandas as pd
import time
import skimage.measure
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy.linalg
from numpy.linalg import inv
import pylab

OPTIMAL = 58.5
EPSILON = 1e-7
BETA = 0.9
reduced_dim = 100
data_squared = None

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
    temp_init_w = np.ones_like(init_w, dtype = np.float64)
    init_w = init_w*temp_init_w
    '''
    # Call individual descent algo - make a note of the obj func, accuracy, time taken
    grad_results = algo(init_w, data, label, grad_step)
    print('Gradient Descent:')
    print(grad_results.time)
    # print(grad_results.objective[0:2])
    newton_results = algo(init_w, data, label, newton_step)
    print('Newton Method:')
    print(newton_results.time)
    print (newton_results.objective[-1])
    # print(newton_results.objective[0:2])
    # print(coord_results.objective[0:2])
    
    UCB_results = UCB1(init_w, data, label, [grad_step, coord_grad_step, newton_step])
    print('MAB with UCB:')
    print(UCB_results.time)
    print (UCB_results.objective)
    '''
    coord_results = algo(init_w, data, label, coord_grad_step)
    print('Co-ordinate Descent:')
    print(coord_results.time)
    print(coord_results.objective[-1])
    '''
    pylab.plot(UCB_results.objective, '--k', label='MAB combined')
    pylab.plot(grad_results.objective, '-b', label = 'Gradient Descent')
    pylab.plot(coord_results.objective,'-g', label = 'Co-ordinate Descent')
    pylab.plot(newton_results.objective, '-r', label = 'Newton Method')
    pylab.title('Objective values')
    pylab.xlabel('Time Step')
    pylab.ylabel('Squared Error Loss')
    pylab.legend()
    #plt.plot(newton_results.objective)
    #plt.plot([1,2,3,4,5,6])
    #print("hi")
    #print(grad_results.objective)
    plt.show()
    print(chosen_arm)
    nn = np.arange(np.asarray(chosen_arm).shape[0])
    pylab.plot(nn,chosen_arm,'-b','*k')
    pylab.xlabel('Time Step')
    pylab.ylabel('Chosen Arm')
    pylab.title('0 = Gradient Descent, 1 = Co-ordinate Descent, 2 = Newton Method')
    plt.show()
    '''
    # Implement UCB2 for MAB 


    # Report Algo used/epoch, net time taken, obj value achieved, net accuracy

def processData():
    # global data_squared
    # first, parse the data
    # /home/oyku/Desktop/Oyku/Convex 10-725/
    mnist = pd.read_csv('mnist_train.csv')

    mnist = pd.DataFrame.as_matrix(mnist)

    # mnist_shuffled = shuffle(mnist)

    label = mnist[:,0]
    data = mnist[:,1::]

    #To use only 0s and 1s - reduced data

    data = data[(label < 2), :] # 8816
    label = label[(label < 2)] # 8816
    
    #If the above is removed - enter dim reduction here

    label = label.reshape([label.shape[0], 1])

    mean = np.mean(data)
    data = data - mean

    cov = np.dot(data.T, data) / data.shape[0]

    U, S, V = np.linalg.svd(cov)

    data = np.dot(data, U[:,:reduced_dim])
    
    data = np.append(data, np.ones([data.shape[0],1]), axis = 1)


    # data_squared = np.sum(data**2, axis = 0)
    
    return (data, label)

def calc_objective(w, data, label):
    result = (label.T - w.dot(data.T))
    result = 0.5*result.dot(result.T)
    return result[0,0]

def calc_grad(w, data, label):
    return -(label.T - w.dot(data.T)).dot(data)

def grad_step(w, data, label):
    step = 1
    current_objective = calc_objective(w, data, label)
    current_grad = calc_grad(w, data, label)
    quadratic_reduction = current_grad.dot(current_grad.T)/2
    while (True):
        temp_w = w - step*calc_grad(w, data, label)
        if (calc_objective(temp_w, data, label) < current_objective - quadratic_reduction*step):
            break
        step = step*BETA

    return w - step*calc_grad(w, data, label)

def calc_grad_i(w, data, label, i):
    return -(label.T - w.dot(data.T)).dot(data[:,i])[0]

def update_w_i(W, data, label, i):
    w = W[:,:]
    step = 1e-6
    current_objective = calc_objective(w, data, label)
    current_grad = calc_grad_i(w, data, label, i)
    return w[:,i] - step*calc_grad_i(w, data, label, i)

"""
def update_w_i(W, data, label, i):
    global BETA
    w = W[:,:]
    step = 1
    current_objective = calc_objective(w, data, label)
    current_grad = calc_grad_i(w, data, label, i)
    quadratic_reduction = current_grad*(current_grad)/2
    temp_w = np.ones_like(w)*w
    while (True):
        temp_w[:,i] = w[:,i] - step*current_grad
        new_objective = calc_objective(temp_w, data, label)
        if (new_objective < current_objective - quadratic_reduction*step):
            break
        step = step*BETA
    return w[:,i] - step*calc_grad_i(w, data, label, i)
"""

"""
def update_w_i(W, Data, label, i):
    global data_squared
    mask = np.ones(W.shape[1], dtype = bool)
    mask[i] = False
    w = W[:,:]
    w_j = w[:,mask]
    data = Data[:,:]
    data_j = data[:,mask]
    temp = ((label.T - w_j.dot(data_j.T)).dot(data[:,i]))
    return temp[0]/(data_squared[i])
"""

def coord_grad_step(W, data, label):
    w = W[:,:]
    for i in range(w.shape[1]):
        w[:,i] = update_w_i(w, data, label, i)
    return w

def inv_newt_hess(data):
    return inv(data.T.dot(data))

def newton_step(W, data, label):
    w = W[:,:]
    l = 0.1
    step = inv_newt_hess(data)
    grad = calc_grad(w, data, label)
    return w - l*grad.dot(step)

def algo(W, data, label, update_rule):
    w = W[:,:]

    objectives = []

    cur_time = time.time()
    objective = calc_objective(w, data, label)
    objectives += [objective]
    while (True):
        w = update_rule(w, data, label)
        new_objective = calc_objective(w, data, label)
        # if (objective-new_objective < EPSILON):
        if (new_objective < OPTIMAL):
            print("breaking")
            break
        objective = new_objective
        objectives += [objective]
    cur_time = time.time() - cur_time

    return Results(objectives, cur_time)


def UCB1(W, data, label, update_rules):
    w = W[:,:]

    objectives = []

    cur_time = time.time()
    objective = calc_objective(w, data, label)
    LARGE = objective

    N = [0 for i in range(len(update_rules))]
    Rewards = [0 for i in range(len(update_rules))]

    objectives += [objective]
    # first, run every algorithm once
    k = 0
    i = 0
    for update_rule in update_rules:
        op_time = time.time()
        w = update_rule(w, data, label)
        op_time = (time.time() - op_time)*100
        k += 1
        new_objective = calc_objective(w, data, label)
        reward = ((objective-new_objective)/(1.0*LARGE))**(1.0/k)
        objective = new_objective
        objectives += [objective]
        reward = reward/(1.0*op_time)
        N[i] += 1
        Rewards[i] = (Rewards[i]*(N[i]-1) + reward)/N[i]
        i += 1
    global chosen_arm
    UCB = [Rewards[j] + np.sqrt(2*np.log(k)/N[j]) for j in range(len(update_rules))]
    chosen_arm = []
    while (True):
        # UCB = [Rewards[j] + np.sqrt(np.log(2*k/N[j])) for j in range(len(update_rules))]
        j = np.argmax(np.asarray(UCB))
        chosen_arm += [j]
        op_time = time.time()
        update_rule = update_rules[j]
        w = update_rule(w, data, label)
        op_time = (time.time() - op_time)*100
        new_objective = calc_objective(w, data, label)
        if (new_objective < OPTIMAL):
        # if (objective - new_objective < EPSILON):
            print("breaking in ucb ", j)
            break
        reward = ((objective-new_objective)/(1.0*LARGE))**(1.0/k)
        objective = new_objective
        objectives += [objective]
        reward = reward/(1.0*op_time)
        k += 1
        N[j] += 1
        Rewards[j] = (Rewards[j]*(N[j]-1) + reward)/N[j]    
        UCB[j] = Rewards[j] + np.sqrt(2*np.log(k)/N[j])
    cur_time = time.time() - cur_time

    return Results(objectives, cur_time)


main()
