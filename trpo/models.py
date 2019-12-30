import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.spaces import Box, Discrete
from torch.distributions import Normal
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

class CategoryPolicy(Net):
    def __init__(self, 
                input_size, 
                output_size, 
                hidden_size=(32,), 
                activation=torch.tanh, 
                output_activation=None):
        super(CategoryPolicy, self).__init__(input_size, 
                                            output_size, 
                                            hidden_size, 
                                            activation, 
                                            output_activation)
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x, a=None, log_p_all_old=None):

        def categorial_kl(log_p0, log_p1):
            return log_p0.exp().mul(log_p0 - log_p1).sum(dim=1).mean()

        logits = Net.forward(self, x)
        logp_all = F.log_softmax(logits, dim=1)
        pi_a = logp_all.exp().multinomial(1).squeeze(dim=1)
        logp_pi_a = logp_all.mul(F.one_hot(pi_a, self.output_size).squeeze().float()).sum(dim=1).mean()

        if a is not None:
            logp_a = logp_all.mul(F.one_hot(a, self.output_size).squeeze().float()).sum(dim=1).mean()
        else:
            logp_a = None

        if log_p_all_old is None:
            logp_all_old = logp_all.clone().detach()
        d_kl = categorial_kl(logp_all_old, logp_all)

        return pi_a, logp_a, logp_pi_a, logp_all, d_kl

class GaussianPolicy(Net):
    def __init__(self, 
                input_size, 
                output_size, 
                hidden_size=(32,), 
                activation=torch.tanh, 
                output_activation=None, 
                EPS=1e-8):
        super(GaussianPolicy, self).__init__(input_size, 
                                            output_size, 
                                            hidden_size, 
                                            activation, 
                                            output_activation)
        self.EPS = EPS
        self.log_std = nn.Parameter(-0.5 * torch.ones(output_size))

    def forward(self, x, a=None, pi_old=None):
        
        def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
            var0, var1 = log_std0.mul(2).exp(), log_std1.mul(2).exp()
            pre_sum = 0.5*(((mu1 - mu0)**2 + var0)/(var1 + self.EPS) - 1) + log_std1 - log_std0
            return pre_sum.sum(dim=1).mean()
        
        mu = Net.forward(self, x)
        std = self.log_std.exp()
        normal = Normal(mu, std)
        pi_a = normal.rsample()
        logp_pi_a = normal.log_prob(pi_a)
        
        if a is not None:
            var = std.pow(2)
            logp_a = -(a - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - self.log_std
            logp_a.sum(1, keepdim=True)
        else:
            logp_a = None

        if pi_old is None:
            mu_old, log_std_old = mu.clone().detach(), self.log_std.clone().detach()
        else:
            mu_old, log_std_old = pi_old.detach()
        d_kl = diagonal_gaussian_kl(mu_old, log_std_old, mu, self.log_std)

        return pi_a, logp_a, logp_pi_a, (mu, self.log_std), d_kl
    
    def to(self, device):
        self.log_std = self.log_std.to(device)
        return super(Net, self).to(device)

class ActorCritic(nn.Module):
    def __init__(self, 
                input_size, 
                action_space, 
                hidden_size=(32,), 
                activation=torch.tanh, 
                output_activation=None, 
                EPS=1e-8, 
                actor=None):
        super(ActorCritic, self).__init__()
        if actor:
            self.actor = actor 
        else:
            if isinstance(action_space, Discrete):
                self.actor = CategoryPolicy(input_size,
                                            action_space.n,
                                            hidden_size,
                                            activation,
                                            output_activation)
            else:
                self.actor = GaussianPolicy(input_size,
                                            action_space.shape[-1], 
                                            hidden_size, 
                                            activation, 
                                            output_activation, 
                                            EPS=1e-8)
        self.critic = Net(input_size, 
                        1, 
                        hidden_size, 
                        activation, 
                        output_activation)

    def forward(self, x, a=None):
        pi_a, logp_a, logp_pi_a, pi, d_kl = self.actor.forward(x, a) 
        v = self.critic.forward(x)
        return pi_a, logp_a, logp_pi_a, pi, d_kl, v

if __name__ == '__main__':
    # action_space = Discrete(2)
    action_space = Box(np.array((-1,-2)), np.array((1, 2)))
    ac = ActorCritic(16, action_space)
    x = torch.randn((4,16))
    # a = torch.randint(2, (4, 1))
    a = torch.randn((4,2))
    pi_a, logp_a, logp_pi_a, logp, d_kl, v = ac.forward(x, a)
    print(pi_a)
    print(logp_a)
    print(logp_pi_a)
    print(logp)
    print(d_kl)
    print(v)
    pass