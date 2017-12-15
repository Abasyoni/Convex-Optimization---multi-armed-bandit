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

OPTIMAL = 0.0

EPSILON = 1e-7
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
    init_w = np.random.rand(1,Features)*1e-5
    temp_init_w = np.ones_like(init_w, dtype = np.float64)
    init_w = init_w*temp_init_w
    

    init_w_gd = init_w * np.ones_like(init_w)
    init_w_n = init_w * np.ones_like(init_w)
    init_w_cd = init_w * np.ones_like(init_w)
    init_w_ucb1 = init_w * np.ones_like(init_w)
    # Call individual descent algo - make a note of the obj func, accuracy, time taken
        #print('MAB with UCB:')
    #UCB_results = UCB1(init_w_ucb1, data, label, [grad_step, coord_grad_step, newton_step])
    #print(UCB_results.time)
    '''
    print('MAB with EXP3:')
    EXP3_results = EXP3(init_w_ucb1, data, label, [grad_step, coord_grad_step, newton_step])
    print(EXP3_results.time)
    #print (UCB_results.objective)
    '''
    print('Gradient Descent:')
    grad_results = algo(init_w_gd, data, label, grad_step)
    print(grad_results.time)
    '''
    print('Newton Method:')
    newton_results = algo(init_w_n, data, label, newton_step)
    print(newton_results.time)
    
    print('Coordinate Descent:')
    coord_results = algo(init_w_cd, data, label, coord_grad_step)
    print(coord_results.time)
    

    #print(coord_results.objective[-1])
       
    #pylab.plot(UCB_results.objective, '--k', label='MAB combined')
    pylab.plot(EXP3_results.store_time, EXP3_results.objective, '--k', label='MAB combined')
    pylab.plot(grad_results.store_time, grad_results.objective, '-b', label = 'Gradient Descent')
    pylab.plot(coord_results.store_time, coord_results.objective,'-g', label = 'Coordinate Descent')
    pylab.plot(newton_results.store_time, newton_results.objective, '-r', label = 'Newton Method')
    pylab.title('Objective values')
    pylab.xlabel('Time Step')
    pylab.ylabel('Squared Error Loss')
    pylab.legend()
    plt.show()
    
    print "gd: ",np.sum((np.asarray(chosen_arm) == 0) + 0)
    print "cd: ",np.sum((np.asarray(chosen_arm) == 1) + 0)
    print "nm: ",np.sum((np.asarray(chosen_arm) == 2) + 0)

    nn = np.arange(np.asarray(chosen_arm).shape[0])
    pylab.scatter(nn,chosen_arm)
    yy = np.array([0,1,2])
    my_yticks = ['Gradient Descent', 'Coordinate Descent', 'Newton Method']
    plt.yticks(yy, my_yticks)
    pylab.xlabel('Time Step')
    pylab.ylabel('Chosen Arm')
    pylab.title('The Descent Algorithm Chosen for Each Time Step')
    plt.show()
    '''
    # Implement UCB2 for MAB 

    # Report Algo used/epoch, net time taken, obj value achieved, net accuracy

def processData():
    # global data_squared
    # first, parse the data
    # /home/oyku/Desktop/Oyku/Convex 10-725/
    mnist = pd.read_csv('/Users/alyazeed/Desktop/Convex/Convex-Optimization---multi-armed-bandit/train_mnist.csv')

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

#def calc_objective(w, data, label):
#    result = (label.T - w.dot(data.T))
#    result = 0.5*result.dot(result.T)
#    return result[0,0]

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def calc_objective(w, data, label):
    w_dot_x = w.dot(data.T)
    #print ('dot ', w_dot_x)
    vfunc = np.vectorize(sigmoid)
    y_hat = vfunc(w_dot_x).T

    L0 = np.log(1.0-y_hat[label < 0.5])
    L1 = np.log(y_hat[label >= 0.5])
    #L = -label.T*(np.log(y_hat)) - (1-label.T)*(np.log(1.0-y_hat))
    #print('y hat, ', y_hat)
    #print('obj ', (1.0*np.sum(L))/(L.shape[0]))
    return (-1.0*np.sum(L0) - 1.0*np.sum(L1))


#def calc_grad(w, data, label):
#    return -(label.T - w.dot(data.T)).dot(data)

def calc_grad(w, data, label):
    w_dot_x = w.dot(data.T)
    vfunc = np.vectorize(sigmoid)
    y_hat = vfunc(w_dot_x)
    #L = np.zeros_like(label)
    #L[label==0] = y_hat[label <= 0.5]
    #L[label==1] = y_hat[label > 0.5]-1.0
    #print ()
    #return ((L.T).dot(data))
    return (((1.0-label.T)*y_hat - label.T*(1.0-y_hat)).dot(data))

def calc_grad_i(w, data, label, i):
    w_dot_x = w.dot(data.T)
    vfunc = np.vectorize(sigmoid)
    y_hat = vfunc(w_dot_x)
    #print ()
    return (((1.0-label.T)*y_hat - label.T*(1.0-y_hat)).dot(data.T[i]))

def inv_newt_hess(w, data):
    w_dot_x = w.dot(data.T)
    vfunc = np.vectorize(sigmoid)
    y_hat = vfunc(w_dot_x)
    d = y_hat*(1.0-y_hat)
    D = np.diag(d.reshape(d.shape[1]))
    return inv((data.T.dot(D)).dot(data))

def update_w_i(W, data, label, i):
    global BETA
    step = 10.0
    w = np.ones_like(W)*W
    current_objective = calc_objective(w, data, label)
    current_grad = calc_grad_i(w, data, label, i)
    quadratic_reduction = -current_grad*(current_grad)/4.0
    temp_w = np.ones_like(w)*w
    while (True):
        temp_w[:,i] = w[:,i] - step*current_grad
        new_objective = calc_objective(temp_w, data, label)
        if (new_objective <= current_objective + quadratic_reduction*step):
            break
        step = step*BETA
    return w[:,i] - step*calc_grad_i(w, data, label, i)


def grad_step(W, data, label):    
    global BETA
    step = 10.0
    w = np.ones_like(W)*W
    current_objective = calc_objective(w, data, label)
    current_grad = calc_grad(w, data, label)
    quadratic_reduction = -current_grad.dot(current_grad.T)/4.0
    temp_w = np.ones_like(w)*w
    while (True):
        temp_w = w - step*current_grad
        new_objective = calc_objective(temp_w, data, label)
        if (new_objective <= current_objective + quadratic_reduction*step):
            break
        step = step*BETA
    return w - step*current_grad

def newton_step(W, data, label):
    global BETA
    w = np.ones_like(W)*W
    current_objective = calc_objective(w, data, label)
    l = 10.0
    step = inv_newt_hess(w, data)
    grad = calc_grad(w, data, label)
    v = -grad.dot(step)
    quadratic_reduction = grad.dot(v.T)/4.0
    while (True):
        temp_w = w + l*v
        if (calc_objective(temp_w, data, label) <= current_objective + l*quadratic_reduction):
            break
        l = l*BETA

    return w + l*v


def coord_grad_step(W, data, label):
    w = np.ones_like(W)*W
    for i in range(w.shape[1]):
        w[:,i] = update_w_i(w, data, label, i)
    return w


def algo(W, data, label, update_rule):
    w = W[:,:]

    objectives = []

    cur_time = time.time()
    objective = calc_objective(w, data, label)
    objectives += [objective]
    while (True):
        w = update_rule(w, data, label)
        new_objective = calc_objective(w, data, label)
        print new_objective
        if (new_objective -OPTIMAL < 0.0):
            print("breaking")
            break
        objective = new_objective
        objectives += [objective]
    cur_time = (time.time() - cur_time)

    store_time = np.arange(len(objectives))*cur_time/(1.0*len(objectives))

    return Results(objectives, cur_time, store_time)


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
        #if (new_objective - objective < EPSILON):
        reward = (((objective-new_objective)/(1.0*LARGE))**(1.0/k))/P[rule_index]
        objective = new_objective
        objectives += [objective]
        if (new_objective - OPTIMAL < EPSILON):
            break
        reward = reward/(1.0*op_time)
        k += 1
        W[rule_index] = W[rule_index]*np.exp(GAMMA*reward/len(update_rules))

    cur_time = time.time() - cur_time

    return Results(objectives, cur_time, store_time)


main()
