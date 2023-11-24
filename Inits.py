import numpy as np

from Utilities import R

class GlorotInit():
    def __init__(self, in_units, out_units, dtype=np.float32):
        self.__in = in_units
        self.__out = out_units
        self.__dtype = dtype
    
    @staticmethod
    def w_init(in_units, out_units, dtype=np.float32):
        div = np.divide(6, in_units+out_units)
        sqrt = np.sqrt(div)
        l, u = -1 * sqrt, sqrt
        return R.uniform((in_units, out_units), l, u)
    
    def __call__(self):
        return self.w_init(self.__in, self.__out, self.__dtype)