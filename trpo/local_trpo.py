import torch
import torch.distributed as dist
from trpo import TRPO

class LocalTRPO(TRPO):
    def __init__(self, 
                actor, 
                critic, 
                value_lr=0.01,
                value_steps_per_update=50,
                cg_steps=10,
                linesearch_steps=10,
                gamma=0.99,
                tau=0.97,
                damping=0.1,
                max_kl=0.01,
                device=torch.device("cpu")):
        super(LocalTRPO, self).__init__(actor, critic, value_lr,
                                        value_steps_per_update,
                                        cg_steps, linesearch_steps,
                                        gamma, tau, damping, max_kl, device)
        self.synchronous_parameters(self.actor)
        self.synchronous_parameters(self.critic)

    def update(self, state, action, reward, next_state, mask):
        actor_loss, critic_loss = super(LocalTRPO, self).update(state, action, reward, next_state, mask)
        self.average_parameters(self.actor)
        self.average_parameters(self.critic)    
        return actor_loss, critic_loss

    def average_parameters(self, model):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        for param in model.parameters():
            dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                param.data /= size
            dist.broadcast(param.data, src=0)
    
    def synchronous_parameters(self, model):
        for param in model.parameters():
            dist.broadcast(param, src=0)