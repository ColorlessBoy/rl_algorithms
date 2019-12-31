import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.optim import Adam

from models import GaussianPolicy, ValueNetwork

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
    def __init__(self, input_size, action_space, args, activation=torch.tanh, output_activation=None):
        self.gamma = args.gamma
        self.tau = args.tau
        self.damping = args.damping
        self.delta = args.delta
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.actor = GaussianPolicy(input_size, 
                                    action_space,
                                    args.hidden_size, 
                                    activation=activation,
                                    output_activation=output_activation).to(self.device)
        self.critic = ValueNetwork(input_size, 
                                    args.hidden_size, 
                                    activation=activation,
                                    output_activation=output_activation).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), args.lr)

    def getGAE(self, state, reward, mask):
        with torch.no_grad():
            value = self.critic(state)
            returns = torch.zeros_like(reward)
            delta = torch.zeros_like(reward)
            advantage = torch.zeros_like(reward)

            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(reward.size(0))):
                returns[i] = reward[i] + self.gamma * prev_return * mask[i]
                delta[i] = reward[i] + self.gamma * prev_value * mask[i] - value.data[i]
                advantage[i] = delta[i] + self.gamma * self.tau * prev_advantage * mask[i]

                prev_return = returns[i, 0]
                prev_value = value.data[i, 0]
                prev_advantage = advantage[i, 0]
        return returns, (advantage - advantage.mean())/advantage.std()

    def get_policy_loss(self, advantage, logp_a, logp_a_prev):
        return -(advantage * torch.exp(logp_a - logp_a_prev)).mean()

    def get_kl_loss(self, state):
        pi = self.actor(state)
        kl_loss = torch.mean(kl_divergence(self.pi_old, pi))
        return kl_loss

    def cg(self, A, b, iters=10, accuracy=1e-10):
        # A is a function: x ==> A(s) = A @ x
        x = torch.zeros_like(b)
        d = b.clone()
        g = -b.clone()
        g_dot_g_old = torch.tensor(1.)
        for _ in range(iters):
            g_dot_g = torch.dot(g, g)
            d = -g + g_dot_g / g_dot_g_old * d
            print(d)
            Ad = A(d)
            alpha = g_dot_g / torch.dot(d, Ad)
            x += alpha * d
            if g_dot_g < accuracy:
                break
            g_dot_g_old = g_dot_g
            g += alpha * Ad
        return x
    
    def linesearch(self, state, action, advantage, fullstep, steps=10):
        prev_params = get_flat_params_from(self.actor)
        # Line search:
        alpha = 1
        for i in range(steps):
            alpha *= 0.5
            new_params = prev_params + alpha * fullstep
            set_flat_params_to(self.actor, new_params)
            kl_loss = self.get_kl_loss(state)
            log_prob_action = self.actor.get_log_prob_action(state, action)
            policy_loss = self.get_policy_loss(advantage, log_prob_action, self.log_prob_action_old)
            if policy_loss < self.policy_loss and kl_loss < self.delta:
                return True, i
        set_flat_params_to(self.actor, prev_params)
        return False, steps
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.sample(state).detach().cpu().numpy()[0]
    
    def update_parameters(self, memory):
        state, action, reward, next_state, mask = memory.sample()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device).unsqueeze(1)

        returns, advantage = self.getGAE(state, reward, mask)

        self.update_actor(state, action, advantage)
        self.update_critic(state, returns)

    def update_actor(self, state, action, advantage):
        value = self.critic(state)        
        log_prob_action = self.actor.get_log_prob_action(state, action)
        self.log_prob_action_old = log_prob_action.clone().detach()
        self.policy_loss = self.get_policy_loss(advantage, log_prob_action, self.log_prob_action_old)

        grads = torch.autograd.grad(self.policy_loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        def get_Hx(x):
            kl_loss = self.get_kl_loss(state)
            grads = torch.autograd.grad(kl_loss, self.actor.parameters(), create_graph=True)
            flat_grad = torch.cat([grad.view(-1) for grad in grads])
            grad_grads = torch.autograd.grad(flat_grad @ x, self.actor.parameters())
            flat_grad_grad = torch.cat([grad_grad.contiguous().view(-1) for grad_grad in grad_grads]).data
            return flat_grad_grad + x * self.damping

        with torch.no_grad():
            self.pi_old = self.actor(state)
        invHg = self.cg(get_Hx, loss_grad, 10)
        lm = torch.sqrt(0.5 * loss_grad @ invHg / self.delta)
        fullstep = invHg / lm

        flag, step = self.linesearch(state, action, advantage, fullstep)
        if flag:
            print("linesearch successes at step {}".format(step))
        else:
            print("linesearch failed")

    def update_critic(self, state, target_value):
        value = self.critic(state)
        value_loss = F.mse_loss(value, target_value)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()