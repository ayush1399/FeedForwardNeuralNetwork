from MyExceptions import *
import numpy as np
from abc import ABC, abstractmethod

shape_mismatch = lambda y_h, y: f"Shape of y_h is {y_h.shape} and y is {y.shape}"

class Loss(ABC):
    def __init__(self):
        self.__loss = None
        
    def __call__(self, y_h, y):
        return self.loss(y_h, y)
    
    @abstractmethod
    def loss(self):
        pass
    
    @property
    def _loss(self):
        return self.__loss
    
    @_loss.setter
    def _loss(self, loss):
        self.__loss = loss

class CrossEntropy(Loss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self._reduction = reduction
    
    def loss(self, y_h, y):
        if y.shape != y_h.shape:
            raise ShapeMismatchError(shape_mismatch(y_h, y))
        else:
            l = -1 * np.log(y_h)
            l = np.multiply(l, y)
            l = np.sum(l)
            self._loss = self._reduce(l)
            return l
        
    def _backward(self, y_h, y):
        return y_h - y
        
    def _reduce(self, loss):
        if self._reduction == 'mean':
            l = loss.mean(axis=0, keepdims=True)
        elif self._reduction == 'sum':
            l = loss.sum(axis=0, keepdims=True)       


class MeanSquareError(Loss):
    def __init__(self):
        super().__init__()
    
    def loss(self, y_h, y):
        if y.shape != y_h.shape:
            raise ShapeMismatchError(shape_mismatch(y_h, y))
        else:        
            e = np.subtract(y_h, y)
            se = np.square(e)
            mse = np.mean(se, axis=0)
            self._loss = mse
            return self._loss / 2
        
    def _backward(self, y_h, y):
        dl = y_h - y
        return dl