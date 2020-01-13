import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from collections import OrderedDict
import math

def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), activation=torch.relu, output_activation=None):
        super().__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        pre_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(pre_size, size))
            pre_size = size
        self.output_layer = nn.Linear(pre_size, output_size)
        self.apply(_weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

# We require that action is in [-1, 1]^n
class PolicyNetwork(Network):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64),
                 activation=torch.relu, output_activation=torch.tanh, 
                 init_std=1.0, max_log_std=2, min_log_std=-20, epsilon=1e-6):

        super(PolicyNetwork, self).__init__(input_size, output_size, hidden_sizes, activation, output_activation)

        self.max_log_std = max_log_std
        self.min_log_std = min_log_std
        self.epsilon = epsilon

        self.log_std = nn.Parameter(torch.full((1, output_size), math.log(init_std)))
        self.apply(_weight_init)

    def forward(self, state):
        mean = super(PolicyNetwork, self).forward(state)
        std = self.log_std.clamp(self.min_log_std, self.max_log_std).exp()
        return Normal(loc=mean, scale=std)
    
    def select_action(self, state):
        pi = self.forward(state)
        y = pi.rsample()
        action = torch.tanh(y)
        log_pi_action = pi.log_prob(y) - torch.log(1 - action.pow(2) + self.epsilon)
        return action, log_pi_action.sum(axis=1, keepdim=True)
    
    def get_log_prob(self, state, action):
        pi = self.forward(state)
        # y = atanh(action)
        y = 0.5 * (torch.log(1 + action + self.epsilon) - torch.log(1 - action + self.epsilon))
        log_pi_action = pi.log_prob(y) - torch.log(1 - action.pow(2) + self.epsilon)
        return log_pi_action.sum(axis=1, keepdim=True)

    def get_mean_action(self, state):
        pi = self.forward(state)
        y = pi.loc
        action = torch.tanh(y)
        return action

class ValueNetwork(Network):
    def __init__(self, input_size, hidden_sizes=(64, 64), \
                activation=torch.relu, output_activation=None):
        super(ValueNetwork, self).__init__(input_size, 1, hidden_sizes,
                                            activation, output_activation)

if __name__ == '__main__':
    state_size = 10
    action_size = 4
    policy = PolicyNetwork(state_size, action_size, (64, 64))
    v_net = ValueNetwork(state_size)
    q_net = QNetwork(state_size, action_size)
    state = torch.randn((2, 10))
    action, _ = policy.select_action(state)
    print("action = ", action)

    print(v_net(state))

    print(q_net(state, action))

    print(policy)

