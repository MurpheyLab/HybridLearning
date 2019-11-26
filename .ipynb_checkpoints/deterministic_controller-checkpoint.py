import torch
import numpy as np


class DeterministicCtrl(object):

    def __init__(self, model, T=20):

        self.model = model
        self.T = T

        self.u = [
            torch.zeros(1,self.model.num_actions)
            for t in range(self.T-1)
        ]

    def __call__(self, state):

        x_t = torch.FloatTensor(state).unsqueeze(0)
        x = []
        reward = []
        for t in range(self.T):
            x.append(x_t.clone())
            x_t, rew = self.model.step(x_t, self.u[t])
            reward.append(rew)

        # stack the inputs and pass through model again
        x_var = torch.cat(x, requires_grad=True)
        df = compute_jacobian(torch.cat(x, requires_grad=True))
