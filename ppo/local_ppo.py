import torch
import torch.distributed as dist
from ppo import PPO

class LocalPPO(PPO):
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
        super(LocalPPO, self).__init__(actor, critic, clip, gamma, 
                                        tau, pi_steps_per_update, 
                                        value_steps_per_update, 
                                        target_kl, device, pi_lr, v_lr)

    def update(self, state, action, reward, next_state, mask):
        actor_loss, value_loss = super(LocalPPO, self).update(state, action, reward, next_state, mask)
        self.average_parameters(self.actor)
        self.average_parameters(self.critic)
        return actor_loss, value_loss

    def average_parameters(self, model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= size
    