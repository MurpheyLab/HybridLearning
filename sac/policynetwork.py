import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-2, log_std_min=-10, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear1 = nn.Linear(num_inputs, hidden_size)
        # self.log_std_linear2 = nn.Linear(hidden_size, hidden_size)
        self.log_std_linear3 = nn.Linear(hidden_size, num_actions)

        self.log_std_linear3.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = torch.sin(self.linear1(state))
        # x = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)

        log_std = state
        log_std = torch.sin(self.log_std_linear1(log_std))
        # log_std = F.relu(self.log_std_linear2(log_std))
        log_std = self.log_std_linear3(log_std)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]
