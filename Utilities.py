from statistics import mean

import sklearn.datasets as skd
import numpy as np

from numpy.random import default_rng
rng = default_rng()
default_dtype = np.float32

set_object_of_shape = lambda s1, s2: f"Setting value of object of shape {s1} with object of shape {s2}."

class ProgressBar:
    def __init__(self, steps):
        self._steps = steps
        self._loss_history = list()
    
    def progress(self, epoch, loss):
        self._loss_history.append(loss)
        l = mean(self._loss_history)
        
        j = len(self._loss_history)
        progress = int(round(((j/self._steps)) * 70, 0))
        
        bar = "="*(progress-1) + ">" + "="*(70-progress)
        
        print(f"Epoch: {epoch:02d}  Step: {j:03d}/{self._steps:03d}  {bar}  Loss: {l:.02f}", end="\r")
        
    def _reset(self):
        self._loss_history = list()
        print()
        

class Datasets():
    @classmethod
    def __cast(cls, X_y, dtype):
        X, y = X_y
        X, y = X.astype(dtype), y.astype(dtype)
        try:
            y.shape[1]
        except IndexError:
            n = y.shape[0]
            y = y.reshape((n, -1))
        return X, y
    
    @classmethod
    def boston(cls, X_y=True, dtype=np.float32):
        return cls.__cast(skd.load_boston(return_X_y=X_y), dtype=dtype)
    
    @classmethod
    def iris(cls, X_y=True, dtype=np.float32):
        return cls.__cast(skd.load_iris(return_X_y=X_y), dtype)
        
    @classmethod
    def diabetes(cls, X_y=True, dtype=np.float32):
        return cls.__cast(skd.load_diabetes(return_X_y=X_y), dtype)
        
    @classmethod
    def digits(cls, X_y=True, dtype=np.float32):
        return cls.__cast(skd.load_digits(return_X_y=X_y), dtype)
        
    @classmethod
    def linnerud(cls, X_y=True, dtype=np.float32):
        return cls.__cast(skd.load_linnerud(return_X_y=X_y), dtype)
        
    @classmethod
    def wine(cls, X_y=True, dtype=np.float32):
        return cls.__cast(skd.load_wine(return_X_y=X_y), dtype)
    
    @classmethod
    def breast_cancer(cls, X_y=True, dtype=np.float32):
        return cls.__cast(skd.load_breast_cancer(return_X_y=X_y), dtype)
    
class R():
    @staticmethod
    def uniform(shape, low=0.0, high=1.0, dtype=default_dtype):
        return rng.uniform(low=low, high=high, size=shape).astype(dtype)
    
    @staticmethod
    def random_indices(high):
        return rng.permutation(high)