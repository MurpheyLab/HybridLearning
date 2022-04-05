import torch
from torch.distributions import Normal
import time
import numpy as np

class StochPolicyWrapper(object):

    def __init__(self, model, policy, samples=10, t_H=10, lam=0.1):


        self.model          = model
        self.policy         = policy
        self.num_actions    = model.num_actions
        self.t_H            = t_H
        self.lam            = lam
        self.samples        = samples

        self.a = torch.zeros(t_H, self.num_actions)

    def reset(self):
        with torch.no_grad():
            self.a.zero_()

    def __call__(self, state):
        # return np.random.normal(0,0.1,size=(2,))
        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.FloatTensor(state.copy()).unsqueeze(0)
            s = s0.repeat(self.samples, 1)
            mu, log_std = self.policy(s)

            # sk, da, log_prob = [], [], []
            sk = torch.zeros(self.t_H,self.samples)
            da = torch.zeros(self.t_H,self.samples,self.num_actions)
            log_prob = torch.zeros(self.t_H,self.samples)

            for t in range(self.t_H):
                pi = Normal(mu,log_std.exp())
                v = pi.sample()
                log_prob[t] = pi.log_prob(v).sum(1)
                # log_prob.append(pi.log_prob(v).sum(1))
                v = torch.tanh(v)
                # v = torch.tanh(pi.sample())
                da_t = v - self.a[t].expand_as(v)
                # log_prob.append(pi.log_prob(da_t).sum(1))
                # log_prob.append(pi.log_prob(self.a[t].expand_as(v)).sum(1))
                da[t] = da_t
                # da.append(da_t)
                # da.append(v)
                s, rew = self.model.step(s, v)
                mu, log_std = self.policy(s)
                sk[t] = rew.squeeze()
                # sk.append(rew.squeeze())
            # sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)
            # log_prob = torch.stack(log_prob)

            # error handling
            for test in [sk,log_prob]:
                if torch.any(torch.isnan(test)):
                    print('got nan in planner')
                    print(sk,log_prob)

            sk = sk + self.lam*log_prob # added 6/2
            # sk = sk.div(self.lam) + log_prob # added 6/7

            sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
            # sk /= torch.norm(sk, dim=1, keepdim=True)
            # log_prob -= torch.max(log_prob, dim=1, keepdim=True)[0] # commented out 6/2
            # log_prob /= (torch.norm(log_prob,dim=1, keepdim=True) + 1e-4)
            # print(sk,log_prob)
            # w = torch.exp(sk.div(self.lam) + log_prob) + 1e-5 # commented out 6/2
            w = torch.exp(sk.div(self.lam)) + 1e-5 # added 6/2
            # w = torch.exp(sk) + 1e-5 # modified 6/7
            w.div_(torch.sum(w, dim=1, keepdim=True))

            for t in range(self.t_H):
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])
                # self.a[t] = torch.mv(da[t].T, w[t])

            return self.a[0].clone().numpy()
