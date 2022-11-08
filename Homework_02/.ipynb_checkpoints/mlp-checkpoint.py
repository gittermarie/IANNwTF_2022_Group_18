import numpy as np

class MLP:
    def __init__(self, layers):
        self.layers = layers # list of layers
        self.output = None
        
    def forward_step(self, net_input):
        for layer in self.layers:
            net_input = layer.forward_step(net_input)
        self.output = net_input
        return self.output
    
    def backpropagation(self, L_a, learning_rate):
        for layer in reversed(self.layers):
            L_a = layer.backward_step(L_a, learning_rate)
            
    def training(self, n_episodes, data, target, learning_rate):
        average_loss = []
        for n in range(n_episodes):
            loss = []
            for i, x in enumerate(data):
                # create prediction
                prediction = (self.forward_step([x])[0])
                # Loss:
                loss.append((0.5*(target[i]-prediction)**2))
                
                # Derivative of Lossfunction w.r.t. the final activation:
                # (final_activation - target)
                L_a = prediction - target[i]
                self.backpropagation(L_a, learning_rate)
            
            # collect average Loss for visualization
            average_loss.append(sum(loss)/len(loss))
            # print(n, ": Loss: ", average_loss[n])
        
        return average_loss