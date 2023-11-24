import numpy as np

class SimpleOptimizer():
    def __init__(self, learning_rate=1e-3):
        self._lr = learning_rate
        
    def update_params(self, network):
        for layer in network:
            layer._W = layer._W - self._lr * layer._dW
            layer._b = layer._b - self._lr * layer._db

class AdamOptimizer():
    def __init__(self, learning_rate=1e-3, beta1=9e-1, beta2=999e-3, eps=1e-8):
        self._lr = learning_rate
        self._b1 = beta1
        self._b2 = beta2
        self._eps = eps
        self.__t = 0
        self.__history = None
        
    @property
    def _t(self):
        return self.__t
    
    @_t.setter
    def _t(self, t):
        self.__t = self._t + 1
        
    def update_params(self, network):
        self._t += 1
        
        for i, layer in enumerate(network):
            if self.__history is None:
                self.__history = dict()
                self.__history[f"{layer}_{i}_dW_m"] = np.zeros(layer._dW.shape)
                self.__history[f"{layer}_{i}_db_m"] = np.zeros(layer._db.shape)
                self.__history[f"{layer}_{i}_dW_v"] = np.zeros(layer._dW.shape)
                self.__history[f"{layer}_{i}_db_v"] = np.zeros(layer._db.shape)            
            
            gt_W, gt_b = layer._dW, layer._db
            
            try:
                _ = self.__history[f"{layer}_{i}_dW_m"]
            except KeyError as err:
                self.__history[f"{layer}_{i}_dW_m"] = np.zeros(layer._dW.shape)
                self.__history[f"{layer}_{i}_db_m"] = np.zeros(layer._db.shape)
                self.__history[f"{layer}_{i}_dW_v"] = np.zeros(layer._dW.shape)
                self.__history[f"{layer}_{i}_db_v"] = np.zeros(layer._db.shape) 
            
            self.__history[f"{layer}_{i}_dW_m"] = self._b1 * self.__history[f"{layer}_{i}_dW_m"] + (1 - self._b1) * gt_W
            self.__history[f"{layer}_{i}_db_m"] = self._b1 * self.__history[f"{layer}_{i}_db_m"] + (1 - self._b1) * gt_b

            self.__history[f"{layer}_{i}_dW_v"] = self._b2 * self.__history[f"{layer}_{i}_dW_v"] + (1 - self._b2) * gt_W * gt_W
            self.__history[f"{layer}_{i}_db_v"] = self._b2 * self.__history[f"{layer}_{i}_db_v"] + (1 - self._b2) * gt_b * gt_b

            dW_m_corr = self.__history[f"{layer}_{i}_dW_m"] / (1 - self._b1 ** self._t)
            db_m_corr = self.__history[f"{layer}_{i}_db_m"] / (1 - self._b1 ** self._t)

            dW_v_corr = self.__history[f"{layer}_{i}_dW_v"] / (1 - self._b2 ** self._t)
            db_v_corr = self.__history[f"{layer}_{i}_db_v"] / (1 - self._b2 ** self._t)

            layer._W = layer._W - self._lr * dW_m_corr / (np.sqrt(dW_v_corr) + self._eps)
            layer._b = layer._b - self._lr * db_m_corr / (np.sqrt(db_v_corr) + self._eps)


class RMSPropOptimizer():
    def __init__(self, learning_rate=1e-3, beta=9e-1, eps=1e-8):
        self._lr = learning_rate
        self._b = beta
        self._eps = eps
        self.__history = None
        
    def update_params(self, network):
        for i, layer in enumerate(network):
            if self.__history is None:
                self.__history = dict()
                self.__history[f"{layer}_{i}_dW_vt"] = np.zeros(layer._dW.shape)
                self.__history[f"{layer}_{i}_db_vt"] = np.zeros(layer._db.shape)
                
            gt_W, gt_b = layer._dW, layer._db
            try:
                _ = self.__history[f"{layer}_{i}_dW_vt"]
            except KeyError as err:
                self.__history[f"{layer}_{i}_dW_vt"] = np.zeros(layer._dW.shape)
                self.__history[f"{layer}_{i}_db_vt"] = np.zeros(layer._db.shape)
                    
                   
            self.__history[f"{layer}_{i}_dW_vt"] = self._b * self.__history[f"{layer}_{i}_dW_vt"] + (1 - self._b) * gt_W * gt_W
            self.__history[f"{layer}_{i}_db_vt"] = self._b * self.__history[f"{layer}_{i}_db_vt"] + (1 - self._b) * gt_b * gt_b

            layer._W = layer._W - self._lr * gt_W / np.sqrt(self.__history[f"{layer}_{i}_dW_vt"] + self._eps)
            layer._b = layer._b - self._lr * gt_b / np.sqrt(self.__history[f"{layer}_{i}_db_vt"] + self._eps)