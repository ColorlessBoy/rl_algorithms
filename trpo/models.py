import torch
import torch.nn as nn
import math

class Net(nn.Module):
    def __init__(self, 
                input_size, 
                output_size, 
                hidden_size=(32,), 
                activation=torch.tanh, 
                output_activation=None):
        super(Net, self).__init__()
        self.hidden_size=hidden_size
        self.activation=activation
        self.output_activation=output_activation

        self.layers = []
        old_size = input_size
        sizes = (*hidden_size, output_size)
        for size in sizes:
            self.layers.append(nn.Linear(old_size, size))
            old_size = size

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

class GaussianPolicy(Net):
    def __init__(self, 
                input_size, 
                output_size, 
                hidden_size=(32,), 
                activation=torch.tanh, 
                output_activation=None):
        super(GaussianPolicy, self).__init__(input_size, 
                                            output_size, 
                                            hidden_size, 
                                            activation, 
                                            output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(output_size))

    def forward(self, x):
        mu = Net.forward(self, x)
        std = self.log_std.exp()
        return mu, std

    def get_log_prob_action(self, action, mu, std):
        log_prob_action = -((action - mu)/(2 * std)).pow(2) - 0.5 * math.log(2 * math.pi) - self.std
        return log_prob_action.sum(1, keepdim=1)

    def to(self, device):
        self.log_std = self.log_std.to(device)
        return super(GaussianPolicy, self).to(device)

class Value(Net):
    def __init__(self,
                input_size,
                hidden_size=(32,),
                activation=torch.tanh,
                output_activation=None):
        super(Value, self).__init__(input_size, 
                                    1, 
                                    hidden_size, 
                                    activation, 
                                    output_activation)
