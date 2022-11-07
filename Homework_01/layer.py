import numpy as np

class Layer:
    def __init__(self, n_inputs, n_units):
        self.bias = np.zeros(n_units)
        self.weights = np.random.rand(n_inputs, n_units)
        # self.layer_input = 
        # self.layer_preactivation = 
        # self.layer_activation =
        
    def show(self):
        print(self.bias)
        print(self.weights)
        
    def forward_step():
        pass
        
    def backward_step():
        pass