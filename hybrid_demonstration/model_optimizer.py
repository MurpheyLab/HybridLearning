import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from .jacobian import jacobian


class ModelOptimizer(object):

    def __init__(self, model, replay_buffer, lr=1e-2, eps=1e-1, lam=0.95):

        # reference the model and buffer
        self.model          = model
        self.replay_buffer  = replay_buffer
        # set the model optimizer
        self.model_optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        # logger
        self._eps = eps
        self._lam = lam
        self.log = {'loss' : [], 'rew_loss': []}

    def update_model(self, batch_size, mini_iter=1):

        for k in range(mini_iter):
            states, actions, rewards, next_states, next_action, done = self.replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            states.requires_grad = True
            next_states = torch.FloatTensor(next_states)
            actions = torch.FloatTensor(actions)
            next_action = torch.FloatTensor(next_action)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            done    = torch.FloatTensor(np.float32(done)).unsqueeze(1)

            pred_mean, pred_std, pred_rew = self.model(states, actions)

            state_dist = Normal(pred_mean, pred_std)

            df = jacobian(pred_mean, states)

            next_vals = self.model.reward_fun(torch.cat([next_states, next_action], axis=1))

            rew_loss = torch.mean(torch.pow((rewards+self._lam*(1-done)*next_vals).detach() - pred_rew,2))
            # rew_loss = torch.mean(torch.pow(rewards - pred_rew,2))

            model_loss = -torch.mean(state_dist.log_prob(next_states)) + self._eps * torch.norm(df, dim=[1,2]).mean()
            # - 1e-3*pred_next_state_dist.entropy().mean()

            loss = 0.5 * rew_loss + model_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
        self.log['loss'].append(loss.item())
        self.log['rew_loss'].append(rew_loss.item())
