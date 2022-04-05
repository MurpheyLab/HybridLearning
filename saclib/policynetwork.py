import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.a_dim = num_actions

        self.mu = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_actions*2)
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #init weights
        self.mu[-1].weight.data.uniform_(-init_w,init_w)
        self.mu[-1].bias.data.uniform_(-init_w,init_w)


        # self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        #
        # self.log_stdl1 = nn.Linear(num_inputs, hidden_size)
        # self.log_stdl2 = nn.Linear(hidden_size, hidden_size)
        #
        # self.mean_linear = nn.Linear(hidden_size, num_actions)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)
        #
        # self.log_std_linear = nn.Linear(hidden_size, num_actions)
        # self.log_std_linear.weight.data.uniform_(-init_w*0., init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w*0., init_w)
        # self.log_std_linear.weight.data.zero_()
        # self.log_std_linear.bias.data.zero_()

    def forward(self, state):
        out = self.mu(state)
        mean, log_std = torch.split(out, [self.a_dim, self.a_dim], dim=1)


        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # # x = torch.sin(self.linear1(state))
        # # x = torch.sin(self.linear2(x))
        #
        # log_std = F.relu(self.log_stdl1(state))
        # log_std = F.relu(self.log_stdl2(log_std))
        #
        # mean    = self.mean_linear(x)
        # # log_std = self.log_std_linear(F.relu(self.linear2(state)))
        # log_std = self.log_std_linear(log_std)
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
        # action = z

        action = action.detach().cpu().numpy()
        return action[0]
