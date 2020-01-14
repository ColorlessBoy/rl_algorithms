import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

from utils import soft_update

class SACNP(object):
    def __init__(self, v_net, q1_net, q2_net, pi_net, vt_net, 
                gamma=0.99, alpha=0.2, lm = 0.1,
                v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = 0.005,
                device=torch.device('cpu')):
        # nets
        self.v_net, self.q1_net, self.q2_net, self.pi_net, self.vt_net = \
            v_net, q1_net, q2_net, pi_net, vt_net

        # hyperparameters
        self.gamma, self.alpha, self.lm = gamma, alpha, lm

        # device
        self.device = device

        # optimization
        self.v_optim  = Adam(self.v_net.parameters(),  lr = v_lr )
        self.q1_optim = Adam(self.q1_net.parameters(), lr = q_lr )
        self.q2_optim = Adam(self.q2_net.parameters(), lr = q_lr)
        self.pi_optim = Adam(self.pi_net.parameters(), lr = pi_lr)
        self.vt_optim = SGD(self.vt_net.parameters(), lr = vt_lr)

    def update(self, batch1, batch2):
        state1  = torch.FloatTensor(batch1[0]).to(self.device)
        action1 = torch.FloatTensor(batch1[1]).to(self.device)
        reward1 = torch.FloatTensor(batch1[2]).to(self.device).unsqueeze(1)
        nstate1 = torch.FloatTensor(batch1[3]).to(self.device)
        mask1   = torch.FloatTensor(batch1[4]).to(self.device).unsqueeze(1)
    
        state2  = torch.FloatTensor(batch2[0]).to(self.device)
        action2 = torch.FloatTensor(batch2[1]).to(self.device)
        reward2 = torch.FloatTensor(batch2[2]).to(self.device).unsqueeze(1)
        nstate2 = torch.FloatTensor(batch2[3]).to(self.device)
        mask2   = torch.FloatTensor(batch2[4]).to(self.device).unsqueeze(1)

        losses, loss_grads = self.get_grads(state1, action1, reward1, nstate1, mask1)
        loss_grad_grads = self.get_grad_grads(state2, action2, reward2, nstate2, mask2, *loss_grads)

        def step(model, model_optim, loss_grad, loss_grad_grad):
            prev_ind = 0
            new_loss_grad = loss_grad + self.lm * loss_grad_grad
            for param in model.parameters():
                flat_size = param.numel()
                param.grad = \
                    new_loss_grad[prev_ind:prev_ind + flat_size].view(param.size())
                prev_ind += flat_size
            model_optim.step()
        
        step(self.q1_net, self.q1_optim, loss_grads[0], loss_grad_grads[0])
        step(self.q2_net, self.q2_optim, loss_grads[1], loss_grad_grads[1])
        step(self.pi_net, self.pi_optim, loss_grads[2], loss_grad_grads[2])
        step(self.v_net,  self.v_optim,  loss_grads[3], loss_grad_grads[3])
        step(self.vt_net, self.vt_optim, loss_grads[4], loss_grad_grads[4])
        return losses

    def get_grads(self, state, action, reward, nstate, mask):

        def get_loss_grad(loss, model):
            grads = torch.autograd.grad(loss, model.parameters())
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
            return loss_grad

        # Q-Loss
        with torch.no_grad():
            q_target = reward + self.gamma * mask * self.vt_net(nstate)
        q1 = self.q1_net(state, action)
        q2 = self.q2_net(state, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q1_loss_grad = get_loss_grad(q1_loss, self.q1_net)
        q2_loss_grad = get_loss_grad(q2_loss, self.q2_net)

        # Pi-Loss
        pi_action, log_pi_action = self.pi_net.select_action(state)
        q = torch.min(self.q1_net(state, pi_action), self.q2_net(state, pi_action))
        pi_loss = (self.alpha * log_pi_action - q).mean()
        pi_loss_grad = get_loss_grad(pi_loss, self.pi_net)

        # V-Loss
        with torch.no_grad():
            v_target = q - self.alpha * log_pi_action
        v = self.v_net(state)
        v_loss = F.mse_loss(v, v_target)
        v_loss_grad = get_loss_grad(v_loss, self.v_net)

        # Vt-Loss
        vt_loss = 0.0
        for v_param, vt_param in zip(self.v_net.parameters(), self.vt_net.parameters()):
            vt_loss += 0.5 * torch.sum((vt_param - v_param.detach())**2)
        vt_loss_grad = get_loss_grad(vt_loss, self.vt_net)
        return (q1_loss.data, q2_loss.data, pi_loss.data, v_loss.data, vt_loss.data), \
            (q1_loss_grad, q2_loss_grad, pi_loss_grad, v_loss_grad, vt_loss_grad)

    def get_grad_grads(self, state, action, reward, nstate, mask,
                q1_loss_grad_old, q2_loss_grad_old, pi_loss_grad_old, v_loss_grad_old, vt_loss_grad_old):

        def get_loss_grad(loss, model):
            grads = torch.autograd.grad(loss, model.parameters(), 
                                        create_graph=True)
            loss_grad = torch.cat([grad.view(-1) for grad in grads])
            return loss_grad

        def get_loss_grad_grad(grad_loss, model, retain_graph=False, allow_unused=False):
            grad_grads = torch.autograd.grad(grad_loss, model.parameters(), \
                                        retain_graph=retain_graph)
            loss_grad_grad = torch.cat([grad_grad.contiguous().view(-1) for grad_grad in grad_grads]).data
            return loss_grad_grad

        # Q-Loss
        q_target = reward + self.gamma * mask * self.vt_net(nstate)
        q1 = self.q1_net(state, action)
        q2 = self.q2_net(state, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q1_loss_grad = get_loss_grad(q1_loss, self.q1_net)
        q2_loss_grad = get_loss_grad(q2_loss, self.q2_net)

        q1_grad_loss = get_loss_grad(q1_loss, self.q1_net) @ q1_loss_grad_old
        q2_grad_loss = get_loss_grad(q2_loss, self.q2_net) @ q2_loss_grad_old


        # Pi-Loss
        pi_action, log_pi_action = self.pi_net.select_action(state)
        q = torch.min(self.q1_net(state, pi_action), self.q2_net(state, pi_action))
        pi_loss = (self.alpha * log_pi_action - q).mean()
        pi_grad_loss = get_loss_grad(pi_loss, self.pi_net) @ pi_loss_grad_old

        # V-Loss
        pi_action2, log_pi_action2 = self.pi_net.select_action(state)
        q = torch.min(self.q1_net(state, pi_action2), self.q2_net(state, pi_action2))
        v_target = q - self.alpha * log_pi_action2
        v = self.v_net(state)
        v_loss = F.mse_loss(v, v_target)
        v_grad_loss = get_loss_grad(v_loss, self.v_net) @ v_loss_grad_old

        # Vt-Loss
        vt_loss = 0.0
        for v_param, vt_param in zip(self.v_net.parameters(), self.vt_net.parameters()):
            vt_loss += 0.5 * torch.sum((vt_param - v_param)**2)
        vt_grad_loss = get_loss_grad(vt_loss, self.v_net) @ vt_loss_grad_old

        grad_loss = q1_grad_loss + q2_grad_loss + pi_grad_loss + v_grad_loss + vt_grad_loss
        print('\n==========================================================================')
        print("grad_loss = ", grad_loss)
        print('==========================================================================\n')
        q1_loss_grad_grad = get_loss_grad_grad(grad_loss, self.q1_net, retain_graph=True)
        q2_loss_grad_grad = get_loss_grad_grad(grad_loss, self.q2_net, retain_graph=True)
        pi_loss_grad_grad = get_loss_grad_grad(grad_loss, self.pi_net, retain_graph=True)
        v_loss_grad_grad  = get_loss_grad_grad(grad_loss, self.v_net, retain_graph=True)
        vt_loss_grad_grad = get_loss_grad_grad(grad_loss, self.vt_net, retain_graph=False)
        
        return q1_loss_grad_grad, q2_loss_grad_grad, pi_loss_grad_grad, v_loss_grad_grad, vt_loss_grad_grad
        
