import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 3
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, batch_size=256, action_lookback=0):
        super(GaussianPolicy, self).__init__()
        # Specifying the Theta Network (will map states to some latent space equal in dimension to action space)
        self.linear_theta_1 = nn.Linear(num_inputs, hidden_dim)
        self.linear_theta_2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear_theta = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear_theta = nn.Linear(hidden_dim, num_actions)

        self.action_lookback = action_lookback

        # Specifying the Phi Network (will adjust the output of the theta network based on previous action)
        # Old stuff
        print("Gaussian Policy ALB:", self.action_lookback)
        self.linear_phi_1 = nn.Linear(num_actions * self.action_lookback, hidden_dim)
        self.linear_phi_2 = nn.Linear(hidden_dim, hidden_dim)
        # To be added on next with rnn implementation:
        # self.lstm = nn.LSTM(num_actions, hidden_dim, 2, batch_first=False)
        # self.hidden = (torch.randn(2, batch_size, hidden_dim), torch.randn(2, batch_size, hidden_dim))

        self.log_scale_linear_phi = nn.Linear(hidden_dim, num_actions)
        self.shift_linear_phi = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        print(self.action_bias, self.action_scale)

    def forward_theta(self, state):
        x = F.relu(self.linear_theta_1(state))
        x = F.relu(self.linear_theta_2(x))
        mean = self.mean_linear_theta(x)
        log_std = self.log_std_linear_theta(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    # Randomly initialize hidden/cell states
    def reset_hidden_state(self, batch_size):
        self.hidden = (torch.randn(self.hidden[0].shape), torch.randn(self.hidden[1].shape))

    def forward_phi(self, prev_actions):
        # Old stuff
        x = F.relu(self.linear_phi_1(prev_actions))
        x = F.relu(self.linear_phi_2(x))

        # Back to old stuff
        shift = self.shift_linear_phi(x)
        shift = torch.clamp(shift, min=-5.0, max=5.0)
        log_scale = self.log_scale_linear_phi(x)
        log_scale = torch.clamp(log_scale, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # Ensures that the scale factor is > 0
        return log_scale, shift

    def sample(self, state, prev_actions):
        # First pass the state through the state network
        mean, log_std = self.forward_theta(state)
        std = log_std.exp()

        # Sample from the base distribution first.
        base_dist = Normal(mean, std)
        base_action = base_dist.rsample()

        # Base distribution
        log_prob = base_dist.log_prob(base_action)

        if self.action_lookback > 0:
            # Use the previous action(s) to shift the mean of the prediction.
            phi_log_scale, phi_shift = self.forward_phi(prev_actions)
            # Now we adjust the mean and std based on the outputs from phi_network
            action = base_action * phi_log_scale.exp() + phi_shift
            # Also want to adjust the mean, so that evaluation mode also works
            mean = mean * phi_log_scale.exp() + phi_shift
            # Affine Transform Scaling
            log_prob -= phi_log_scale
        else:
            action = base_action

        # Tanh the action
        tanh_action = torch.tanh(action)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - tanh_action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_ret = torch.tanh(mean) * self.action_scale + self.action_bias
        action_ret = tanh_action * self.action_scale + self.action_bias

        return action_ret, log_prob, mean_ret

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

    
    
    
    
"""
TODO: Adjust ARRL to use a Deterministic Policy
"""
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)