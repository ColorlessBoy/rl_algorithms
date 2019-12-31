import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=(32, 32,), activation=torch.tanh, output_activation=None):
        super(Network, self).__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.layers = []
        sizes = (input_size, *hidden_size, output_size)
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            self.add_module('layer{0}'.format(i), self.layers[-1])
        self.apply(weights_init_)

    def forward(self, state):
        x = self.layers[0](state)
        for layer in self.layers[1:]:
            x = layer(self.activation(x))
        if self.output_activation:
            x = self.output_activation(x)
        return x

class ValueNetwork(Network):
    def __init__(self, input_size, hidden_size=(32, 32,), activation=torch.tanh, output_activation=None):
        super(ValueNetwork, self).__init__(input_size, 1, hidden_size, activation, output_activation)

class GaussianPolicy(Network):
    def __init__(self, input_size, action_space, hidden_size=(32,32,), activation=torch.tanh, output_activation=None):
        super(GaussianPolicy, self).__init__(input_size, hidden_size[-1], hidden_size[:-1], activation, activation)
        
        self.mean_linear = nn.Linear(hidden_size[-1], action_space.shape[0])
        self.log_std_linear = nn.Linear(hidden_size[-1], action_space.shape[0])
        self.apply(weights_init_)
        self.output_activation = output_activation

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.layers[0](state)
        for layer in self.layers[1:]:
            x = layer(self.activation(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        if self.output_activation:
            mean = self.output_activation(mean)
            log_std = self.output_activation(log_std)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return Normal(mean, torch.exp(log_std))
    
    def get_log_prob_action(self, state, action):
        # action encode
        action = (action - self.action_bias) / self.action_scale
        pi = self.forward(state)
        return pi.log_prob(action).sum(1, keepdim=True)

    def sample(self, state):
        pi = self.forward(state)
        action = pi.rsample()
        action.clamp_(-1, 1)
        # action decode
        action = action * self.action_scale + self.action_bias
        return action
    
    def to(self, device):
        self.action_bias = self.action_bias.to(device)
        self.action_scale = self.action_scale.to(device)
        return super(GaussianPolicy, self).to(device)