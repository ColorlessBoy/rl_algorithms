import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
import torch.distributed as dist
from ppo import PPO

class GlobalPPO(PPO):
    def __init__(self, 
                actor, 
                critic, 
                clip=0.2, 
                gamma=0.995,
                tau=0.99,
                pi_steps_per_update=80, 
                value_steps_per_update=80,
                target_kl=0.01,
                device=torch.device("cpu"),
                pi_lr=3e-4,
                v_lr=1e-3):
        super(GlobalPPO, self).__init__(actor, critic, clip, gamma, 
                                        tau, pi_steps_per_update, 
                                        value_steps_per_update, 
                                        target_kl, device, pi_lr, v_lr)
        self.synchronous_parameters(self.actor)
        self.synchronous_parameters(self.critic)

    def average_variables(self, variables, size=None):
        if size == None:
            size = float(dist.get_world_size())
        dist.all_reduce(variables, op=dist.ReduceOp.SUM)
        variables /= size

    def average_parameters_grad(self, model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            self.average_variables(param.grad.data, size)
    
    def synchronous_parameters(self, model):
        for param in model.parameters():
            dist.broadcast(param, src=0)
    
    def update_actor(self, state, action, advantage):
        #update actor network
        old_pi = self.actor.get_detach_pi(state)
        log_action_probs = self.actor.get_log_prob(state, action)
        old_log_action_probs = log_action_probs.clone().detach()
        actor_loss = 0.0
        
        for i in range(self.pi_steps_per_update):
            ratio = torch.exp(log_action_probs - old_log_action_probs)
            ratio2 = ratio.clamp(1 - self.clip, 1 + self.clip)
            actor_loss = -torch.min(ratio * advantage, ratio2 * advantage).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.average_parameters_grad(self.actor)
            self.actor_optim.step()

            pi = self.actor.get_detach_pi(state)
            kl = kl_divergence(old_pi, pi).sum(axis=1).mean()
            self.average_variables(kl)

            if kl > self.target_kl:
                break

            log_action_probs = self.actor.get_log_prob(state, action)
        
        return actor_loss
    
    
    def update_critic(self, state, target_value):
        # update critic network
        critic_loss = 0.0
        for _ in range(self.value_steps_per_update):
            value = self.critic(state)
            critic_loss = F.mse_loss(value, target_value)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.average_parameters_grad(self.critic)
            self.critic_optim.step()
        return critic_loss