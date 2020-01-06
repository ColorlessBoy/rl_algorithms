import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, kl_divergence

class PPO(object):
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
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=pi_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=v_lr)
        self.clip = torch.tensor(clip, device=device)
        self.gamma = torch.tensor(gamma)
        self.tau = torch.tensor(tau)
        self.pi_steps_per_update = torch.tensor(pi_steps_per_update, device=device)
        self.value_steps_per_update = torch.tensor(value_steps_per_update, device=device)
        self.target_kl = torch.tensor(target_kl, device=device)
        self.device = device
    
    def getGAE(self, state, reward, mask):
        # On CPU.
        with torch.no_grad():
            value = self.critic(state).cpu()
            returns = torch.zeros_like(reward)
            delta = torch.zeros_like(reward)
            advantage = torch.zeros_like(reward)

            prev_return = torch.tensor(0.0, device=self.device)
            prev_value = torch.tensor(0.0, device=self.device)
            prev_advantage = torch.tensor(0.0, device=self.device)
            for i in reversed(range(reward.size(0))):
                returns[i, 0] = reward[i, 0] + self.gamma * prev_return * mask[i, 0]
                delta[i, 0] = reward[i, 0] + self.gamma * prev_value * mask[i, 0] - value[i, 0]
                advantage[i, 0] = delta[i, 0] + self.gamma * self.tau * prev_advantage * mask[i, 0]

                prev_return = returns[i, 0]
                prev_value = value[i, 0]
                prev_advantage = advantage[i, 0]
        return returns, (advantage - advantage.mean())/advantage.std()

    def update(self, state, action, reward, next_state, mask):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1) # cpu
        # next_state = torch.FloatTensor(next_state).to(self.device)
        mask = torch.FloatTensor(mask).unsqueeze(1) # cpu

        # Get generalized advantage estimation
        # and get target value
        target_value, advantage = self.getGAE(state, reward, mask) # On CPU
        actor_loss = self.update_actor(state, action, advantage.to(self.device))
        critic_loss = self.update_critic(state, target_value.to(self.device))
        return actor_loss, critic_loss
        
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
            self.actor_optim.step()

            pi = self.actor.get_detach_pi(state)
            kl = kl_divergence(old_pi, pi).sum(axis=1).mean()
            if kl > self.target_kl:
                print("Upto target_kl at Step {}".format(i))
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
            self.critic_optim.step()
        return critic_loss
        

