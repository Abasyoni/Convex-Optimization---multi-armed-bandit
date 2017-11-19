
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.utils import shuffle


get_ipython().magic(u'matplotlib inline')



get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[2]:


mnist = pd.read_csv('/home/oyku/Desktop/Oyku/Convex 10-725/train_mnist.csv')

mnist = pd.DataFrame.as_matrix(mnist)

mnist_shuffled = shuffle(mnist)

print mnist.shape


# In[3]:


s = plt.imshow(mnist[0,1::].reshape([28,28]))


# In[4]:


label = mnist[:,0]
data = mnist[:,1::]
print data.shape
print np.sum((label == 0) + 0)
print np.sum((label == 1) + 0)

new_data = data[(label < 2), :]
new_label = label[(label < 2)]
print new_data.shape
print new_label.shape


# In[5]:


X = new_data
y = new_label.reshape([new_label.shape[0], 1])
y[y == 0] = -1
print X.shape
print y.shape


# In[6]:


#Centralize the data
for i in xrange(X.shape[1]):
    X[:,i] = X[:,i] - np.mean(X[:,i])
X = np.append(X, np.ones([X.shape[0],1]), axis = 1)
print X.shape


# In[7]:


s = plt.imshow(X[0,0:-1].reshape([28,28]))


# In[8]:


#Train and Validation sets
train = X[0:6100,:]
train_y = y[0:6100,:]

valid = X[6100:7458,:]
valid_y = y[6100:7458,:]

test = X[7458:8816,:]
test_y = y[7458:8816,:]

print train.shape
print valid.shape
print test.shape


# In[9]:


#model parameters
n = train.shape[0] #number of examples in train
d = train.shape[1] #number of features in train

#C = 10
# epsilon = 0.5
# t = 5
alpha = 0.5

beta = 0.9 #backtracking step decrease rate
#t_mu = 1.2 #scaling factor of t for the barrier method


# In[10]:


initial_w = np.random.rand(1,d)*1e-3


# # Gardient Descent

# In[11]:


w = initial_w[:,:]

runs = 10
epoch = 10
loss_store = np.zeros([1,runs*epoch])
accuracy = np.zeros([1,epoch])


count = 0
for e in xrange(epoch):
    for r in xrange(runs):
        step = 1

        dw = -(train_y.T - w.dot(train.T)).dot(train)
        dw /= n

        #Backtracking
        temp_w = w - step*dw

        loss = 0.5*(train_y.T - w.dot(train.T)).dot(train_y - w.dot(train.T).T)
        loss_right = loss + step*alpha*loss.T.dot(loss)
        loss_right /= n
        loss_left = 0.5*(train_y.T - temp_w.dot(train.T)).dot(train_y - temp_w.dot(train.T).T)
        loss_left /= n

        while loss_left > loss_right:
            step = beta*step
            temp_w = w - step*dw

            loss = 0.5*(train_y.T - w.dot(train.T)).dot(train_y - w.dot(train.T).T)
            loss_right = loss + step*alpha*loss.T.dot(loss)
            loss_right /= n
            loss_left = 0.5*(train_y.T - temp_w.dot(train.T)).dot(train_y - temp_w.dot(train.T).T)
            loss_left /= n

        w = temp_w

        predict_valid = w.dot(valid.T).T
        predicted = (predict_valid > 0) + 0
        predicted[predicted == 0] = -1
        loss_store[:,count] = loss_left
        print "---------- epoch: ", "e,  iteration: ", count , " ----------"
        print "Loss: ", loss_store[:,count]
        print "Step: ", step
        count += 1

    accuracy[:,e] = np.sum((predicted == valid_y) + 0)/(1.0*valid_y.shape[0]) * 100
    print
    print "Validation accuracy: ", accuracy[:,e]


# In[12]:


predict_test = w.dot(test.T).T
predicted_test = (predict_test > 0) + 0
predicted_test[predicted_test == 0] = -1
accuracy_test = np.sum((predicted_test == test_y) + 0)/(1.0*test_y.shape[0]) * 100
print accuracy_test


# In[13]:


plt.plot(accuracy.T, label='MSE')
plt.title('Accuracy' )
plt.xlabel('epoch number')
plt.ylabel('MSE')
plt.legend()
plt.show()


# In[14]:


print accuracy


# # Coordinate Descent

# In[15]:


w_c = initial_w[:,:]

runs = 5
epoch = 5

loss_store = np.zeros([1,runs*epoch*w_c.shape[1]])
accuracy_c = np.zeros([1,epoch])

step = 1e-6
count = 0
for e in xrange(epoch):
    for r in xrange(runs):
        for i in xrange(w_c.shape[1]):

            dw = -(train_y.T - w_c.dot(train.T)).dot(train)
            dw /= n
            
            w_c[:,i] = w_c[:,i] - step*dw[:,i]
            #Backtracking
            
            loss = 0.5*(train_y.T - w_c.dot(train.T)).dot(train_y - w_c.dot(train.T).T)
            loss /= n
            
        predict_valid = w_c.dot(valid.T).T
        predicted = (predict_valid > 0) + 0
        predicted[predicted == 0] = -1
        loss_store[:,count] = loss
        print "-------------- ", r, " --------------"
        print "Loss: ", loss_store[:,count]
        count += 1

    accuracy_c[:,e] = np.sum((predicted == valid_y) + 0)/(1.0*valid_y.shape[0]) * 100
    print
    print "Validation accuracy: ", accuracy_c[:,e]


# In[16]:


predict_test_c = w_c.dot(test.T).T
predicted_test_c = (predict_test_c > 0) + 0
predicted_test_c[predicted_test_c == 0] = -1
accuracy_test_c = np.sum((predicted_test_c == test_y) + 0)/(1.0*test_y.shape[0]) * 100
print accuracy_test_c

