import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from models import GaussianPolicy, Value

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

class trpo(object):
    def __init__(self, input_size, action_space, args, activation, output_activation):
        self.gamma = args.gamma
        self.tau = args.tau
        self.damping = args.damping
        self.delta = args.delta
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.actor = GaussianPolicy(input_size, 
                                    action_space.shape[-1], 
                                    args.hidden_size, 
                                    activation=activation,
                                    output_activation=output_activation)
        self.critic = Value(input_size, 
                            args.hidden_size, 
                            activation=activation,
                            output_activation=output_activation)
        self.critic_optim = Adam(self.critic.parameters(), args.lr)

    def getGAE(self, reward, value, gamma, tau):
        # (batch_size, shape)
        delta = reward[:, :-1] + gamma * value[:, 1:] - value[:, :-1]
        print("delta = ", delta)
        advantage = reward.clone()
        for i in reversed(range(delta.shape[-1])):
            advantage[0, i] = delta[0, i] + gamma * tau * advantage[0, i+1]    
        return (advantage - advantage.mean())/advantage.std()

    def get_policy_loss(self, advantage, logp_a, logp_a_prev):
        return -advantage * torch.exp(logp_a - logp_a_prev)
    
    def get_gaussian_kl(self, mu, std, mu_old, std_old):
        # return KL(pi_old || pi)
        # p ~ N(mu_1, std_1)
        # q ~ N(mu_2, std_2)
        # KL(p || q) = log(std_2 / std_1) + (var_1 + (mu_1 - mu_2)**2) - 1) / (2 * var_2)
        var, var_old = std**2, std_old**2
        kl = std.log() - std_old.log() + (var_old + (mu - mu_old)**2 - 1) / (2 * var)
        return kl.sum(axis=1).mean()
    
    def get_Hx(self, x):
        grads = torch.autograd.grad(self.kl_loss, self.actor.parameters(), create_graph=True)
        flat_grad = torch.cat([grad.view(-1) for grad in grads])
        grad_grads = torch.autograd.grad(flat_grad @ x, self.actor.parameters())
        flat_grad_grad = torch.cat([grad_grad.contiguous().view(-1) for grad_grad in grad_grads]).data
        return flat_grad_grad + x * self.damping

    def cg(self, A, b, iters, accuracy=1e-10):
        # A is a function: x ==> A(s) = A @ x
        x = torch.zeros_like(b)
        d = b.clone()
        g = -b
        g_dot_g_old = 1
        for _ in range(iters):
            g_dot_g = np.dot(g, g)
            d = -g + g_dot_g / g_dot_g_old * d
            alpha = g_dot_g / torch.dot(d, A(d))
            x += alpha * d
            if g_dot_g < accuracy:
                break
            g_dot_g_old = g_dot_g
            g = A(x) - b
        return x
    
    def parameters(self, memory):
        state, action, reward, next_state = memory.sample()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        self.update_actor(state, action, reward, next_state)
        self.update_critic(state, action, reward, next_state)

    def update_actor(self, state, action, reward, next_state):
        value = self.critic(state)        
        advantage = self.getGAE(reward, value, self.gamma, self.tau)

        # Just for GaussianPolicy
        policy_mean, policy_std = self.actor(state)
        log_prob_action = self.actor.get_log_prob_action(action, policy_mean, policy_std)
        log_prob_action_old = log_prob_action.clone().detach()
        self.policy_loss = self.get_policy_loss(advantage, log_prob_action, log_prob_action_old)

        grads = torch.autograd.grad(self.policy_loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        policy_mean_old = policy_mean.clone().detach()
        policy_std_old = policy_std.clone().detach()
        self.kl_loss = self.get_gaussian_kl(policy_mean, policy_std, policy_mean_old, policy_std_old)
        
        invHg = self.cg(self.get_Hx, loss_grad, 10)
        lm = torch.sqrt(0.5 * loss_grad @ invHg / self.delta)
        fullstep = invHg / lm

        prev_params = get_flat_params_from(self.actor)
        # Line search:
        alpha = 1
        for _ in range(10):
            alpha = 0.5 * alpha
            new_params = prev_params + alpha * fullstep
            set_flat_params_to(self.actor, new_params)
            policy_mean, policy_std = self.actor(state)
            log_prob_action = self.actor.get_log_prob_action(action, policy_mean, policy_std)
            kl_loss = self.get_gaussian_kl(policy_mean, policy_std, policy_mean_old, policy_std_old)
            policy_loss = self.get_policy_loss(advantage, log_prob_action, log_prob_action_old)
            if policy_loss < self.policy_loss and kl_loss < self.delta:
                break

    def update_critic(self, state, action, reward, next_state):
        value = self.critic(state)        
        target_value = reward.clone()
        target_value[:, :-1] += self.gamma * value[:, 1:] - value[:, :-1] 
        value_loss = F.mse(value, target_value)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()