import numpy as np

class Layer:
    def __init__(self, n_inputs, n_units):
        self.bias = np.zeros(n_units)
        self.weights = np.random.rand(n_units, n_inputs)
        self.layer_input = None 
        self.layer_preactivation = None
        self.layer_activation = None
        
    def show(self):
        print(self.bias)
        print(self.weights)
        
    def forward_step(self, layer_input):
        self.layer_input = layer_input
        
        # input is multiplied by weights and bias is added:
        self.layer_preactivation = self.weights @ self.layer_input + self.bias
        
        # ReLU activation:
        self.layer_activation = np.maximum(self.layer_preactivation, 0)
        
        return self.layer_activation
        
    def backward_step(self, L_a, learning_rate):
        # How does the Loss change w.r.t. the activation? (L_a)
        # This is calculated outside of the Layers as a first input and then passed on
        
        # transpose for multiply:
        L_a = np.transpose(L_a)
        
        # How does the activation change w.r.t. the preactivation? (a_p)
        # This is just the derivative of our activation -> ReLU
        a_p = [1 if x>0 else 0 for x in self.layer_preactivation]
        
        # How does the preactivation change w.r.t the weights? (p_w)
        # This is what we use to update our weights
        p_w = np.transpose([self.layer_input])
        weight_gradient = p_w @ np.multiply(a_p, L_a)
        
        bias_gradient = np.multiply(a_p, L_a)
        self.weights = self.weights - learning_rate * np.transpose(weight_gradient)
        self.bias = self.bias - learning_rate * bias_gradient[0]
        
        # How does the preactivation change w.r.t the inputs? (p_i)
        # This is what we need to determine our return that we feed into subsequent/ previous Layer
        input_gradient = np.multiply(a_p, L_a) @ np.transpose([self.weights])
        return input_gradient
        