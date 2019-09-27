#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import h5py
from scipy import ndimage
import sklearn.datasets
import math

get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


#sigmoid activation unit calculation
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A,Z

#relu activation unit calculation
def relu(Z):
    A = np.maximum(0,Z)
    return A,Z


# In[51]:


#deravatives calculations
def relu_backward(dA, cache): 
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0 
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s) 
    return dZ


# In[52]:


#rendom initialisation. HE for relu  ans XAVIER for tanh activation unit.
def initialize_parameters_deep(layer_dims,activation_unit="relu"):
    parameters = {}
    L = len(layer_dims)         
    
    if activation_unit == "relu":
        for l in range(1, L):                    #he is used for relu
            parameters['W' + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l-1]))*(np.sqrt(2/layer_dims[l-1])) 
            parameters['b' + str(l)] =  np.zeros((layer_dims[l], 1))
    else:
        for l in range(1, L):                       #xavier is used for tanh
            parameters['W' + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l-1]))*(np.sqrt(1/layer_dims[l-1])) 
            parameters['b' + str(l)] =  np.zeros((layer_dims[l], 1))
     
    return parameters


# In[53]:


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
   
    if activation == "sigmoid":
     
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache

#cache --> linear  --> A,W,b   and activation  --> Z, these informations are needed in backpropagation steps.
def L_model_forward(X, parameters):
  
    caches = []
    A = X
    L = len(parameters) // 2              
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
    
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    return AL, caches


# In[54]:


#cost computation with regularisation factor.  if do not want to use ragularisation  set lambd=0.
def compute_cost(AL,Y,parameters,lambd=0):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    
    L=len(parameters)//2
    regularisation_cost=0
    for i in range(L):
        regularisation_cost+=np.sum(np.square(parameters["W"+str(i+1)]))
    
    return cost+(lambd/(2*m))*regularisation_cost


# In[55]:


def linear_backward(dZ, cache,lambd):
   
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)+(lambd/m)*W #due to regularisation
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
  
    return dA_prev, dW, db

def linear_activation_backward(dA, cache,lambd, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches,lambd):
   
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
  
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,lambd, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,lambd, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# In[56]:


#GRADIENT CHECKING UNIT.

#convert to vector.
def to_vector(gradients):
    count = 0
    for key in gradients.keys():
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

#convert to dictionary.
def to_dictionary(theta,dims):
    parameters = {}
    p=0
    for i in range(1,len(dims)):
        parameters["W"+str(i)] = theta[p:p+dims[i]*dims[i-1]].reshape((dims[i],dims[i-1]))
        p+=dims[i]*dims[i-1]
        parameters["b"+str(i)] = theta[p:p+dims[i]].reshape((dims[i],1))
        p+=dims[i]
   
    return parameters

'''
theta=to_vector(parameters)
print(theta.shape)
paramters=to_dictionary(theta,layers_dims)
print(parameters["W4"].shape)
'''

#gradient check function. very slow if neural networrk has huge no. of neurons
def grad_check(params,grads,X,Y,dims,epsilon=1e-7):
    paramv=to_vector(params)
    gradv=to_vector(grads)
    n=paramv.shape[0]
    
    J_plus=np.zeros((n,1))
    J_minus=np.zeros((n,1))
    grad_approx=np.zeros((n,1))
    
    for i in range(n):
        theta_plus=np.copy(paramv)
        theta_plus[i][0]+=epsilon
        AL, _ =L_model_forward(X,to_dictionary(theta_plus,dims))
        J_plus[i]=compute_cost(AL,Y,params)
        
        theta_minus=np.copy(paramv)
        theta_minus[i][0]-=epsilon
        AL, _ =L_model_forward(X,to_dictionary(theta_minus,dims))
        J_minus[i]=compute_cost(AL,Y,params)
        
        grad_approx[i]=(J_plus[i]-J_minus[i])/(2*epsilon)
        
    numerator=np.linalg.norm(gradv-grad_approx,keepdims=True)
    denominator=np.linalg.norm(gradv,keepdims=True)+np.linalg.norm(grad_approx,keepdims=True)
        
    diff=numerator/denominator
    if diff>2e-7:
        print("There is an error in gradient calculations.")
    else:
        print("Everything is ok.")


# In[57]:


#GENERATE RANDOM MINI BATCHES.
def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]                  
    mini_batches = []
        
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_adam(parameters) :
    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
        s["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        s["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
    return v, s

#ADAM UPDATION OF PARAMETERS.
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
   
    L = len(parameters) // 2                 
    v_corrected = {}                        
    s_corrected = {}                        
    
    for l in range(L):
        
        v["dW" + str(l+1)] = beta1*v["dW"+str(l+1)]+(1-beta1)*grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta1*v["db"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]
       
        v_corrected["dW" + str(l+1)]= v["dW" + str(l+1)] /(1-np.power(beta1,t))
        v_corrected["db" + str(l+1)]= v["db" + str(l+1)] /(1-np.power(beta1,t))
       
        s["dW" + str(l+1)] = beta2*s["dW"+str(l+1)]+(1-beta2)*grads["dW"+str(l+1)]*grads["dW"+str(l+1)]
        s["db" + str(l+1)] = beta2*s["db"+str(l+1)]+(1-beta2)*grads["db"+str(l+1)]*grads["db"+str(l+1)]
      
        s_corrected["dW" + str(l+1)]=  s["dW" + str(l+1)] /(1-np.power(beta2,t))
        s_corrected["db" + str(l+1)]=  s["db" + str(l+1)] /(1-np.power(beta2,t))
      
        parameters["W" + str(l+1)] -= learning_rate*(v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)])+epsilon))
        parameters["b" + str(l+1)] -= learning_rate*(v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)])+epsilon))

    return parameters, v, s


# In[67]:


def model(X, Y,layers_dims,learning_rate = 0.0007,batch_size=64,beta1=0.9,beta2=0.999,epsilon=1e-8,epochs=200,
                  print_cost=True,lambd=0):
   
    costs = []                       
    parameters = initialize_parameters_deep(layers_dims)
    t=0
    m=X.shape[1]
    
    v,s=initialize_adam(parameters)
    for i in range(epochs):
        minibatches=random_mini_batches(X,Y,batch_size)
        cost_total=0
        
        for minibatch in minibatches:
            (miniX,miniY)=minibatch
            AL, caches = L_model_forward(miniX,parameters)
        
            cost_total = compute_cost(AL,miniY,parameters,lambd)
      
            grads = L_model_backward(AL,miniY,caches,lambd)
            t=t+1
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8)
            
        avg_cost=cost_total/m
        #if i==1000 or i== 2000:                             #for gradient checking
         #   grad_check(parameters,grads,X,Y,layers_dims)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, avg_cost))
        if print_cost and i % 100 == 0:
            costs.append(avg_cost)
        
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[74]:


def load_dataset():
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) 
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y

X,Y=load_dataset()
layers_dims = [X.shape[0], 20, 7, 5, 1]        #architechture of our neural network.
parameters = model(X,Y,layers_dims)           #train model and get parameters.


# In[75]:


#predict function for binary classifier.
def predict(X,Y,parameters):
    m=X.shape[1]
    p=np.zeros((1,m))
    
    probas,caches=L_model_forward(X,parameters)
    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i]=1
        else:
            p[0,i]=0
    c=0
    for i in range(Y.size):
        if Y[0][i]==p[0][i]:
            c+=1
    print("accuracy is %f" %((c/Y.size)*100))
    return p


# In[76]:


predict(X,Y,parameters)


# In[ ]:




