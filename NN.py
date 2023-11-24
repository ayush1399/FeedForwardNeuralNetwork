from Utilities import ProgressBar, R
from Optim import SimpleOptimizer

class FFNN():
    def __init__(self, network, loss):
        self._network = network
        self._loss = loss()
        
    def _forward(self, X):
        z = X
        for layer in self._network:
            z = layer(z)
        
        return z
    
    def _backward(self, y_h, y):
        cache = self._loss._backward(y_h, y)
#         print("LAST", cache.shape)

        for i, layer in enumerate(reversed(self._network)):
            if i:
                cache = layer._backward(cache)
            else:
                cache = layer._backward(cache, last=True)
    
    def __call__(self, X):
        return self._forward(X)
            
    def train(self, X, y, batch_size=None, epochs=32, optimizer=None, random_state=0):
        
        if optimizer is None:
            optimizer = SimpleOptimizer()
        
        N = X.shape[0]
        batch_size = N if batch_size is None else batch_size
        
        steps = self.__calc_steps(N, batch_size)
        self._pb = ProgressBar(steps)
        
        for i in range(epochs):
            self.__epoch(X, y, N, batch_size, steps, i+1, optimizer)
            
    def __epoch(self, X, y, N, batch_size, steps, epoch, optimizer):
        idxs = R.random_indices(N)
        for i in range(steps):
            batch_idxs = self.__calc_batchidxs(i, steps, idxs, batch_size)
            X_batch, y_batch = X[batch_idxs, :], y[batch_idxs, :]
            
            y_h = self._forward(X_batch)
            loss = self._loss(y_h, y_batch)
            
            self._backward(y_h, y_batch)
            optimizer.update_params(self._network)
            
            self._pb.progress(epoch, loss.item())
        self._pb._reset()

            
    def __calc_batchidxs(self, i, steps, idxs, batch_size):
        if i+1 == steps:
            batch_idxs = idxs[i*batch_size: ]
        else:
            batch_idxs = idxs[i*batch_size: (i+1)*batch_size]
        return batch_idxs
    
    def __calc_steps(self, N, batch_size):
        steps = N/batch_size
        if steps == int(steps):
            steps = int(steps)
        else:
            steps = int(steps) + 1
        return steps