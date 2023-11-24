from MyExceptions import *
import numpy as np
from Inits import *
from Utilities import *
from ActivationFunctions import ReLU

class DenseLayer():
    def __init__(self, in_units, out_units, activation=ReLU,
                 weight_init=GlorotInit, dtype=np.float32):
        
        self.__units = (in_units, out_units)
        
        self.__W = weight_init(in_units, out_units, dtype=dtype)()
        self.__b = np.ones((1, out_units), dtype=dtype)
        
        self.__dW = np.empty((in_units, out_units), dtype=dtype)
        self.__db = np.empty((1, out_units), dtype=dtype)
        
        self.__X, self.__y = None, None

        self.__activation = activation(out_units, dtype)
        
        self.__gradset = False
        
    def __call__(self, vals):
        return self._forward(vals)
    
    def _forward(self, X):
        self._X = X
        self._y = X @ self._W + self._b
        return self.__activation(self._y)
    
    
    @property
    def _X(self):
        return self.__X
    
    @_X.setter
    def _X(self, X):
        self.__X = np.copy(X)
    
    @property
    def _y(self):
        return self.__y
    
    @_y.setter
    def _y(self, y):
        self.__y = np.copy(y)
    
    @property
    def _W(self):
        return self.__W
    
    @_W.setter
    def _W(self, W):
        if self._W.shape != W.shape:
            raise ShapeMismatchError(set_object_of_shape(self._W.shape, W.shape))
        np.copyto(self.__W, W)#, casting='no')
    
    @property
    def _b(self):
        return self.__b
    
    @_b.setter
    def _b(self, b):
        if self._b.shape != b.shape:
            raise ShapeMismatchError(set_object_of_shape(self._b.shape, b.shape))
        np.copyto(self.__b, b)#, casting='no')
        
    @property
    def _dW(self):
        return self.__dW
    
    @_dW.setter
    def _dW(self, dW):
        if dW.shape == self._W.shape:
            self.__dW = dW
        else:
            raise ShapeMismatchError
    
    @property
    def _db(self):
        return self.__db
    
    @_db.setter
    def _db(self, db):
        if db.shape == self._b.shape:
            self.__db = db
        else:
            raise ShapeMismatchError
        
    @property
    def _gradset(self):
        return self.__gradset
        
    @_gradset.setter
    def _gradset(self, gradset):
        if gradset is False or gradset is True:
            self.__gradset = gradset
        else:
            raise ValueError("Only True/False values are allowed.")    
            
    def _backward(self, cache, last=False):
        if last:
            dz = cache
        else:
            dzn, dwn = cache['dzn'], cache['dwn']
#             print("ELSE", dzn.shape, dwn.T.shape, self.__activation._backward().shape)
            dz = dzn @ dwn.T * self.__activation._backward()

        m = self._X.shape[0]
            
#         print(self._X.T.shape, dz.shape)
#         print(dz.sum(axis=0, keepdims=True).shape)
        self._dW = (1/m) * self._X.T @ dz
        self._db = (1/m) * dz.sum(axis=0, keepdims=True)
        
        cache = {
            'dzn': self._y,
            'dwn': self._W
        }
        
#         print("*"*70, end="\n\n\n")
        return cache

    def __repr__(self):
        return f"DenseLayer: {self.__units[0]} -> {self.__units[0]} Activation: {self.__activation}"