import torch
import torch.distributed as dist
from trpo import TRPO

class DMTRPO(TRPO):
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
        super(DMTRPO, self).__init__(actor, critic, value_lr,
                                    value_steps_per_update,
                                    cg_steps, linesearch_steps,
                                    gamma, tau, damping, max_kl, device)
        self.synchronous_parameters(self.actor)
        self.synchronous_parameters(self.critic)

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

    def update(self, state, action, reward, next_state, mask):
        actor_loss, critic_loss = super(DMTRPO, self).update(state, action, reward, next_state, mask)
        self.average_parameters(self.actor)
        self.average_parameters(self.critic)    
        return actor_loss, critic_loss

    def update_actor(self, state, action, advantage):
        log_prob_action = self.actor.get_log_prob(state, action)
        self.log_prob_action_old = log_prob_action.clone().detach()
        self.actor_loss_old = self.get_actor_loss(advantage, log_prob_action, self.log_prob_action_old)

        grads = torch.autograd.grad(self.actor_loss_old, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        # Synchronous here.
        self.average_parameters(loss_grad)

        self.pi_old = self.actor.get_detach_pi(state)

        def get_Hx(x):
            kl_loss = self.get_kl_loss(state)
            grads = torch.autograd.grad(kl_loss, self.actor.parameters(), create_graph=True)
            flat_grad = torch.cat([grad.view(-1) for grad in grads])
            grad_grads = torch.autograd.grad(flat_grad @ x, self.actor.parameters())
            flat_grad_grad = torch.cat([grad_grad.contiguous().view(-1) for grad_grad in grad_grads]).data
            return flat_grad_grad + x * self.damping

        invHg = self.cg(get_Hx, loss_grad, self.cg_steps)
        lm = torch.sqrt(0.5 * loss_grad @ invHg / self.max_kl)
        fullstep = invHg / lm

        rank = dist.get_rank()
        flag, step, actor_loss = self.linesearch(state, action, advantage, fullstep)
        if flag:
            print("Rank{}: linesearch successes at step {}".format(rank, step))
        else:
            print("Rank{}: linesearch failed".format(rank))
        return actor_loss