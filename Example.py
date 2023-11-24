#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np

from sklearn.preprocessing import OneHotEncoder


# In[8]:


import Optim as Opt
from Utilities import *
from MyExceptions import *
from LossFunctions import *
from Inits import *
from Layers import *
from ActivationFunctions import *

from NN import FFNN


# In[35]:


X, y = Datasets.diabetes()
y = y.reshape((442, -1))
X.shape, y.shape


# In[37]:


epochs = 5000


NN = FFNN([
        DenseLayer(10, 20, ReLU),
        DenseLayer(20, 40, Sigmoid),
        DenseLayer(40, 50, Tanh),
        DenseLayer(50, 1, Linear)
    ],
    MeanSquareError
)

adam = Opt.RMSPropOptimizer()

NN.train(X, y, 32, epochs, optimizer=adam)

