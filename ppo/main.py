import gym
import numpy as np
import torch

from models import PolicyNetwork, ValueNetwork
from ppo import PPO

import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', 
            ('state', 'action', 'reward', 'next_state', 'mask'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

class EnvSampler(object):
    def __init__(self, env, max_episode_step=1000):
        self.env = env
        self.max_episode_step = max_episode_step
        self.env_init()
        self.action_scale = (env.action_space.high - env.action_space.low)/2
        self.action_bias = (env.action_space.high + env.action_space.low)/2
    
    # action_encode and action_decode project action into [-1, 1]^n
    def action_encode(self, action):
        return (action - self.action_bias)/self.action_scale
    
    def action_decode(self, action_):
        return action_ * self.action_scale + self.action_bias
    
    def env_init(self):
        self.state = self.env.reset()
        self.done = False
        self.episode_step = 1
    
    def __call__(self, get_action, batch_size):
        # I suggest batch_size to be the multiple of max_episode_step.
        memory = Memory()
        batch_reward = 0.0
        for _ in range(batch_size):
            action_ = get_action(self.state)
            action =self.action_decode(action_)
            next_state, reward, self.done, _ = self.env.step(action) 
            # The env will automatically clamp action into [action_space.low, action_space.high]^n
            batch_reward += reward
            mask = 1.0 if not self.done else 0.0
            memory.push(self.state, action_, reward, next_state, mask)
            self.state = next_state
            self.episode_step += 1
            if self.done or self.episode_step > self.max_episode_step:
                self.env_init()
        return batch_reward, memory.sample()


# The properties of args:
# 1. env_name (default = 'HalfCheetah-v2')
# 2. device (default = "cuda:0")
# 3. seed (default = 1)
# 4. hidden_sizes (default = (64, 32))
# 5. episodes (default = 100. Not the number of trajectories, but the number of batches.)
# 6. max_episode_step (default = 1000)
# 7. batch_size (default = 4000)
# 8. gamma (default = 0.99)
# 9. tau (default = 0.97)
# 10. clip (default = 0.2)
# 11. max_kl (default =  0.01)
# 12. pi_steps_per_update (default = 80) 
# 13. value_steps_per_update (default = 80)
# 14. pi_lr (default = 3e-4)
# 15. value_lr (default = 1e-3)
def main(args):
    env = gym.make(args.env_name)
    device = torch.device(args.device)

    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # 2.Create actor, critic, EnvSampler() and PPO.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = PolicyNetwork(state_size, action_size, hidden_sizes=args.hidden_sizes)
    critic = ValueNetwork(state_size, hidden_sizes=args.hidden_sizes)
    env_sampler = EnvSampler(env, args.max_episode_step)
    ppo = PPO(actor, 
              critic, 
              clip=args.clip, 
              gamma=args.gamma, 
              tau=args.tau, 
              target_kl=args.target_kl, 
              device=device,
              pi_steps_per_update=args.pi_steps_per_update,
              value_steps_per_update=args.value_steps_per_update,
              pi_lr=args.pi_lr,
              v_lr=args.value_lr)

    # 3.Start training.
    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor.select_action(state)
        return action.detach().cpu().numpy()[0]

    total_step = 0
    for episode in range(1, args.episodes+1):
        episode_reward, samples = env_sampler(get_action, args.batch_size)
        actor_loss, value_loss = ppo.update(*samples)
        yield episode*args.batch_size, episode_reward, actor_loss, value_loss

if __name__ == "__main__":
    Args = namedtuple('Args',
                ('env_name', 
                 'device', 
                 'seed', 
                 'hidden_sizes', 
                 'episodes', 
                 'max_episode_step', 
                 'batch_size', 
                 'gamma', 
                 'tau', 
                 'clip', 
                 'target_kl',
                 'pi_steps_per_update',
                 'value_steps_per_update',
                 'pi_lr',
                 'value_lr'))
    args = Args('HalfCheetah-v2',   # env_name
                'cuda:0',           # device
                12345,              # seed
                (64, 64),           # hidden_sizes
                10000,              # episodes
                1000,               # max_episode_step
                1000,               # batch_size
                0.99,               # gamma
                0.97,               # tau
                0.2,                # clip
                0.02,               # target_kl
                80,                 # pi_steps_per_update
                80,                 # value_steps_per_update
                3e-4,               # pi_lr
                1e-3)               # value_lr
    for step, reward, actor_loss, value_loss in main(args):
        reward = reward * args.max_episode_step / args.batch_size
        print('Step {}: Reward = {}, actor_loss = {}, value_loss = {}'.format(step, reward, actor_loss, value_loss))