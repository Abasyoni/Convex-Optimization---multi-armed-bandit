import numpy as np
import pandas as pd
import time
import skimage.measure
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy.linalg
from numpy.linalg import inv
import pylab
from numpy.random import choice

OPTIMAL = 38.5#38.4854465414
EPSILON = 1e-4
BETA = 0.9
GAMMA = 0.5
SCALE_TIME = 200 #120
reduced_dim = 100
data_squared = None

class Results(object):
    def __init__(self, objective, time, store_time):
        self.objective = objective
        self.time = time
        self.iter = len(objective)
        self.store_time = store_time


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
    

    init_w_gd = init_w * np.ones_like(init_w)
    init_w_n = init_w * np.ones_like(init_w)
    init_w_cd = init_w * np.ones_like(init_w)
    init_w_ucb1 = init_w * np.ones_like(init_w)
    init_w_exp3 = init_w * np.ones_like(init_w)
    init_w_b = init_w * np.ones_like(init_w)
    # Call individual descent algo - make a note of the obj func, accuracy, time taken
    
    print('MAB with UCB1:')
    UCB_results = UCB1(init_w_ucb1, data, label, [grad_step, coord_grad_step, newton_step, batch_grad_step])
    print(UCB_results.time)
    
    print('Batch Gradient Descent:')
    batch_results = algo(init_w_b, data, label, batch_grad_step)
    print(batch_results.time)
    
    print('MAB with EXP3:')
    EXP3_results = EXP3(init_w_exp3, data, label, [grad_step, coord_grad_step, newton_step, batch_grad_step])
    print(EXP3_results.time)
    
    print('Gradient Descent:')
    grad_results = algo(init_w_gd, data, label, grad_step)
    print(grad_results.time)

    print('Newton Method:')
    newton_results = algo(init_w_n, data, label, newton_step)
    print(newton_results.time)

    print('Coordinate Descent:')
    coord_results = algo(init_w_cd, data, label, coord_grad_step)
    print(coord_results.time)
    
    pylab.plot(UCB_results.objective, '--c', label='MAB with UCB(1)')
    pylab.plot(EXP3_results.store_time, EXP3_results.objective, '--k', label='MAB with EXP(3)')
    pylab.plot(grad_results.store_time, grad_results.objective, '-b', label = 'Gradient Descent')
    pylab.plot(coord_results.store_time, coord_results.objective,'-g', label = 'Coordinate Descent')
    pylab.plot(newton_results.store_time, newton_results.objective, '-r', label = 'Newton Method')
    pylab.plot(batch_results.store_time, batch_results.objective, '-m', label = 'Mini-batch Gradient Descent')
    
    pylab.title('Objective values')
    pylab.xlabel('Time')
    pylab.ylabel('Squared Error Loss')
    pylab.legend()
    plt.show()
    
    print "gd: ",np.sum((np.asarray(chosen_arm) == 0) + 0)
    print "cd: ",np.sum((np.asarray(chosen_arm) == 1) + 0)
    print "nm: ",np.sum((np.asarray(chosen_arm) == 2) + 0)
    
    nn = np.arange(np.asarray(chosen_arm).shape[0])
    pylab.scatter(nn,chosen_arm)
    yy = np.array([0,1,2,3])
    #yy = np.array([0,1,2])
    my_yticks = ['Gradient Descent', 'Coordinate Descent', 'Newton Method', 'Mini-Batch']
    plt.yticks(yy, my_yticks)
    pylab.xlabel('Time Step')
    pylab.ylabel('Chosen Arm')
    pylab.title('The Descent Algorithm Chosen for Each Time Step using EXP(3)')
    plt.show()


    nn_UCB = np.arange(np.asarray(chosen_arm_UCB).shape[0])
    pylab.scatter(nn_UCB,chosen_arm_UCB)
    #yy = np.array([0,1,2,3])
    #yy = np.array([0,1,2])
    #my_yticks = ['Gradient Descent', 'Coordinate Descent', 'Newton Method', 'Mini-Batch']
    plt.yticks(yy, my_yticks)
    pylab.xlabel('Time Step')
    pylab.ylabel('Chosen Arm')
    pylab.title('The Descent Algorithm Chosen for Each Time Step using UCB(1)')
    plt.show()
    # Implement UCB2 for MAB 

    # Report Algo used/epoch, net time taken, obj value achieved, net accuracy

def processData():
    # global data_squared
    # first, parse the data
    # /home/oyku/Desktop/Oyku/Convex 10-725/
    mnist = pd.read_csv('~/Desktop/Oyku/Convex 10-725/repo/Convex-Optimization---multi-armed-bandit/train_mnist.csv')

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

def batch_grad_step(w, data, label):
    global BETA
    step = 1.0
    frac = np.round(data.shape[0]*0.3).astype(int)
    index = np.random.choice(data.shape[0], frac, replace=False)
    batch = data[index,:]
    label_b = label[index,:]
    current_objective = calc_objective(w, batch, label_b)
    current_grad = calc_grad(w, batch, label_b)
    quadratic_reduction = current_grad.dot(current_grad.T)/2
    """
    backtracking = True
    if np.isnan(current_objective):
        step = 0
        backtracking = False
    """
    while (True):
        temp_w = w - step*calc_grad(w, batch, label_b)
        if (calc_objective(temp_w, batch, label_b) <= current_objective - quadratic_reduction*step + EPSILON/10):
            break
        step = step*BETA
    return w - step*calc_grad(w, batch, label_b)


def calc_grad_i(w, data, label, i):
    return -(label.T - w.dot(data.T)).dot(data[:,i])[0]

def update_w_i(W, data, label, i):
    w = W[:,:]
    step = 1e-10
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
    l = 0.001
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
        if (new_objective - OPTIMAL < EPSILON):
            print("breaking")
            break
        objective = new_objective
        objectives += [objective]
    cur_time = (time.time() - cur_time)

    store_time = np.arange(len(objectives))*cur_time/(1.0*len(objectives))

    return Results(objectives, cur_time, store_time)


def UCB1(W, data, label, update_rules):
    w = W[:,:]

    objectives = []
    store_time = [0.0]
    cur_time = time.time()
    objective = calc_objective(w, data, label)
    LARGE = objective

    N = [0 for i in range(len(update_rules))]
    Rewards = [0 for i in range(len(update_rules))]

    objectives += [objective]
    # first, run every algorithm once
    k = 0
    i = 0
    global chosen_arm_UCB
    chosen_arm_UCB = []
    for update_rule in update_rules:
        op_time = time.time()
        w = update_rule(w, data, label)
        op_time = (time.time() - op_time)*SCALE_TIME
        store_time = [store_time[-1] + op_time/SCALE_TIME]
        k += 1
        chosen_arm_UCB += [i]
        new_objective = calc_objective(w, data, label)
        reward = ((objective-new_objective)/(1.0*LARGE))**(1.0/k)
        objective = new_objective
        objectives += [objective]
        reward = reward/(1.0*op_time)
        N[i] += 1
        Rewards[i] = (Rewards[i]*(N[i]-1) + reward)/N[i]
        i += 1
    UCB = [Rewards[j] + np.sqrt(2*np.log(k)/N[j]) for j in range(len(update_rules))]
    
    while (True):
        j = np.argmax(np.asarray(UCB))
        chosen_arm_UCB += [j]
        op_time = time.time()
        update_rule = update_rules[j]
        w = update_rule(w, data, label)
        op_time = (time.time() - op_time)*SCALE_TIME
        new_objective = calc_objective(w, data, label)
        if (new_objective - OPTIMAL < EPSILON):
        #if (objective - new_objective < EPSILON):
            print("breaking in ucb ", j)
            break
        objective = new_objective
        objectives += [objective]
        #OLD REWARD
        #reward = ((objective-new_objective)/(1.0*LARGE))**(1.0/k)
        #reward = reward/(1.0*op_time)
        
        #NEW REWARD
        reward = ((objective-new_objective)/(1.0*LARGE) + 1)/2.0
        reward = reward/(1.0*op_time)

        k += 1
        N[j] += 1
        Rewards[j] = (Rewards[j]*(N[j]-1) + reward)/N[j]
        #UCB = [Rewards[j] + np.sqrt(2*np.log(k/N[j])) for j in range(len(update_rules))]  
        UCB[j] = Rewards[j] + np.sqrt(2*np.log(k)/N[j])
    cur_time = time.time() - cur_time

    return Results(objectives, cur_time, store_time[1::])

def EXP3(W, data, label, update_rules):
    w = W[:,:]

    objectives = []
    store_time = [0.0]
    cur_time = time.time()
    objective = calc_objective(w, data, label)
    LARGE = objective

    W = [1.0 for i in xrange(len(update_rules))]
    P = [1.0/len(update_rules) for i in xrange(len(update_rules))]

    objectives += [objective]
    # first, run every algorithm once
    global chosen_arm
    chosen_arm = []

    k = 1
    while (True):
        for i in xrange(len(P)):
            P[i] = (1-GAMMA)*W[i]/sum(W) + GAMMA/len(update_rules)

        rule_index = choice(range(len(update_rules)), p=P)
        update_rule = update_rules[rule_index]
        chosen_arm += [rule_index]
        op_time = time.time()
        w = update_rule(w, data, label)
        op_time = (time.time() - op_time)*SCALE_TIME
        store_time += [store_time[-1] + op_time/SCALE_TIME]
        new_objective = calc_objective(w, data, label)
        #OLD REWARD
        #reward = (((objective-new_objective)/(1.0*LARGE))**(1.0/k))/P[rule_index]
        #reward = reward/(1.0*op_time)

        #NEW REWARD
        reward = ((objective-new_objective)/(1.0*LARGE) + 1)/2.0
        reward = reward/(1.0*op_time)
        reward = reward/(1.0*P[rule_index])

        objective = new_objective
        objectives += [objective]
        if (new_objective - OPTIMAL < EPSILON):
            break
        k += 1
        W[rule_index] = W[rule_index]*np.exp(GAMMA*reward/len(update_rules))

    cur_time = time.time() - cur_time

    return Results(objectives, cur_time, store_time)


main()