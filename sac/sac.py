import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

class SAC(object):
    def __init__(self, v_net, q_net, q2_net, pi_net, vt_net, 
                gamma=0.99, alpha=0.2,
                v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = 0.02,
                device=torch.device('cpu')):
        # nets
        self.v_net, self.q_net, self.q2_net, self.pi_net, self.vt_net = \
            v_net.to(device), q_net.to(device), q2_net.to(device), pi_net.to(device), vt_net.to(device)

        # hyperparameters
        self.gamma, self.alpha = gamma, alpha

        # device
        self.device = device

        # optimization
        self.v_optim  = Adam(self.v_net.parameters(),  lr = v_lr )
        self.q_optim  = Adam([self.q_net.parameters(), self.q2_net.parameters()],  lr = q_lr )
        self.pi_optim = Adam(self.pi_net.parameters(), lr = pi_lr)
        self.vt_optim = SGD(self.vt_net.parameters(), lr = vt_lr)

    def update(self, state, action, reward, nstate, mask):
        state  =  torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        nstate = torch.FloatTensor(nstate).to(self.device)
        mask   = torch.FloatTensor(mask).to(self.device).unsqueeze(1)

        q_loss  = self.update_q_net(state, action, reward, nstate, mask)

        pi_action, log_pi_action = self.pi_net.select_action(state)
        q = torch.min(self.q_net(state, pi_action), self.q2_net(state, pi_action))

        pi_loss = self.update_pi_net(log_pi_action, q)
        v_loss, v = self.update_v_net(state, log_pi_action, q)
        vt_loss = self.update_vt_net(state, v)

        return q_loss, pi_loss, v_loss, vt_loss
    
    def update_q_net(self, state, action, reward, nstate, mask):
        with torch.no_grad():
            q_target = reward + self.gamma * mask * self.vt_net(nstate)
        q = self.q_net(state, action)
        q2 = self.q2_net(state, action)
        q_loss = F.mse_loss(q, q_target) + F.mse_loss(q2, q_target)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        return q_loss
    
    def update_pi_net(self, log_pi_action, q):
        pi_loss = (self.alpha * log_pi_action - q).mean()

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        return pi_loss
    
    def update_v_net(self, state, log_pi_action, q):
        pi_action, log_pi_action = self.pi_net.select_action(state)
        v_target = q - self.alpha * log_pi_action
        v = self.v_net(state)
        v_loss = F.mse_loss(v, v_target)

        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        return v_loss, v
    
    def update_vt_net(self, state, vt_target):
        vt = self.vt_net(state)
        vt_loss = F.mse_loss(vt, vt_target)

        self.vt_optim.zero_grad()
        vt_loss.backward()
        self.vt_optim.step()

        return vt_loss
