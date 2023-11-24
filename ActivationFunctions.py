from MyExceptions import *

from Utilities import set_object_of_shape
import numpy as np

from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    def __init__(self, units, dtype=np.float32):
        self.__y = None
        self.__dy = None
        
    def __call__(self, vals):
        return self._forward(vals)
    
    def _forward(self, X):
        self._y = self.__activation(X)
        return self._y
    
    @abstractmethod
    def __activation(self, X):
        pass
        
    @property
    def _y(self):
        return self.__y
    
    @_y.setter
    def _y(self, y):
#         if self._y.shape != y.shape:
#             raise ShapeMismatchError(set_object_of_shape(self._y.shape, y.shape))
#         np.copyto(self.__y, y, ='no')
        self.__y = np.copy(y)
    
    @property
    def _dy(self):
        return self.__dy
    
    @_dy.setter
    def _dy(self, dy):
        if dy is None:
            self.__dy = None
        elif dy.shape != self._y.shape:
            raise ShapeMismatchError(set_object_of_shape(self._y.shape, dy.shape))
        else:
            self.__dy = np.copy(dy)

            
class Sigmoid(ActivationFunction):
    def __init__(self, units, dtype=np.float32):
        super().__init__(units, dtype)
    
    def _ActivationFunction__activation(self, X):
        op = np.exp(-X)
        op = np.add(op, 1)
        op = np.divide(1, op)
        self._y = np.copy(op)
        return self._y
    
    def _backward(self, cache=None):
        self._dy = np.multiply(self._y, 1 - self._y)
        return self._dy
            
    def __repr__(self):
        return f"Sigmoid"    


class Linear(ActivationFunction):
    def __init__(self, units, dtype=np.float32):
        super().__init__(units, dtype)
    
    def _ActivationFunction__activation(self, X):
        self._y = np.copy(X)
        return self._y
    
    def _backward(self, cache=None):
        self._dy = np.ones(self._y.shape)
        return self._dy
            
    def __repr__(self):
        return f"Linear"
    
class Tanh(ActivationFunction):
    def __init__(self, units, dtype=np.float32):
        super().__init__(units, dtype)
    
    def _ActivationFunction__activation(self, X):
        emx, epx = np.exp(-X), np.exp(X)
        op = np.divide(epx - emx, epx + emx)
        self._y = op
        return self._y
    
    def _backward(self, cache=None):
        self._dy = 1 - np.multiply(self._y, self._y)
        return self._dy

    def __repr__(self):
        return f"Tanh"
    
class ReLU(ActivationFunction):
    def __init__(self, units, dtype=np.float32):
        super().__init__(units, dtype)
    
    def _ActivationFunction__activation(self, X):
        op = np.copy(X)
        op[X < 0] = 0
        self._y = op
        return self._y
    
    def _backward(self, cache=None):
        self._dy = np.copy(self._y)
        self._dy[self._dy > 0] = 1
        return self._dy
       
    def __repr__(self):
        return f"ReLU"

class Softmax(ActivationFunction):
    def __init__(self, units, dtype=np.float32):
        super().__init__(units, dtype)
    
    def _ActivationFunction__activation(self, X):
        epx = np.exp(X)
        self._y = np.divide(epx, epx.sum(axis=1).reshape((-1, 1)))
        return self._y
    
    def _backward(self, cache=None):
        self._dy = self._y * (cache['grad'] -(cache['grad'] * self._y).sum(axis=1)[:,None])
        return self._dy
    
    def __repr__(self):
        return f"Softmax"