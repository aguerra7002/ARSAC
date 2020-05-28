import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 5
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
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                action_lookback=0, use_prev_states=False, use_gated_transform=False, ignore_scale=False, hidden_dim_base=256 ):
        super(GaussianPolicy, self).__init__()
        # Specifying the Theta Network (will map states to some latent space equal in dimension to action space)
        self.linear_theta_1 = nn.Linear(num_inputs, hidden_dim_base)
        #self.linear_theta_2 = nn.Linear(hidden_dim, hidden_dim) # For 2 hidden layers; maybe make this an option later.
        self.mean_linear_theta = nn.Linear(hidden_dim_base, num_actions)
        self.log_std_linear_theta = nn.Linear(hidden_dim_base, num_actions)

        # How far back we look with the phi network
        self.action_lookback = action_lookback
        # Do we use prev actions and states? Or just prev actions
        self.use_prev_states = use_prev_states
        # Do we use inverse autoregressive transform
        self.use_gated_transform = use_gated_transform
        # Do we use a scaling factor of 1?
        self.ignore_scale = ignore_scale

        if use_prev_states:
            phi_input_dim = (num_inputs + num_actions) * self.action_lookback
        else:
            phi_input_dim = num_actions * self.action_lookback

        if action_lookback > 0:
            self.linear_phi_1 = nn.Linear(phi_input_dim, hidden_dim)
            self.linear_phi_2 = nn.Linear(hidden_dim, hidden_dim)

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

    def forward_theta(self, state):
        x = F.relu(self.linear_theta_1(state))
        #x = F.relu(self.linear_theta_2(x))
        mean = self.mean_linear_theta(x)
        log_std = self.log_std_linear_theta(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def forward_phi(self, prev_states, prev_actions):

        if self.use_prev_states:
            # Concatenate the previous states to the previous actions input to get new input.
            inp = torch.cat((prev_states, prev_actions), 1)
        else:
            inp = prev_actions
        x = F.relu(self.linear_phi_1(inp))
        x = F.relu(self.linear_phi_2(x))

        if self.use_gated_transform:
            # Using inverse ar flows
            sigma = torch.sigmoid(self.log_scale_linear_phi(x))
            m = self.shift_linear_phi(x)
            #m = torch.clamp(m) # TODO: See if we need to clamp?
            return m, sigma
        else:
            shift = self.shift_linear_phi(x)
            shift = torch.clamp(shift, min=-5.0, max=5.0)
            log_scale = self.log_scale_linear_phi(x)
            log_scale = torch.clamp(log_scale, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # Ensures that the scale factor is > 0
            return shift, log_scale


    def sample(self, state, prev_states, prev_actions, return_distribution=False, random_base=False):
        # First pass the state through the state network
        mean, log_std = self.forward_theta(state)
        std = log_std.exp()

        if random_base:
            mean = torch.zeros(mean.shape)
            std = torch.ones(std.shape)

        # Sample from the base distribution first.
        base_dist = Normal(mean, std)
        base_action = base_dist.rsample()

        # Base distribution
        log_prob = base_dist.log_prob(base_action)

        if self.action_lookback > 0:
            m, sigma = self.forward_phi(prev_states, prev_actions)
            if self.use_gated_transform:
                # Inverse ar transform
                action = sigma * base_action + (1 - sigma) * m
                log_prob -= sigma.log() # Convert back to log so prb is additive
                # For logging
                ascle = sigma
                ashft = m
            else:
                # If we are here, we are doing the standard autoregressive transform
                # Now we adjust the mean and std based on the outputs from phi_network
                if self.ignore_scale:
                    # If we only want to incorporate the shift execute this code
                    action = base_action + m
                    mean = mean + m
                    # For logging
                    ascle = torch.ones(base_action.shape)
                    ashft = m
                else:
                    action = base_action * sigma.exp() + m
                    # Also want to adjust the mean, so that evaluation mode also works
                    mean = mean * sigma.exp() + m
                    # Affine Transform Scaling
                    log_prob -= sigma
                    # For logging
                    ascle = sigma.exp()
                    ashft = m

        else:
            action = base_action
            # For logging
            ascle = torch.ones(action.shape)
            ashft = torch.zeros(action.shape)

        # Tanh the action
        tanh_action = torch.tanh(action)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - tanh_action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_ret = torch.tanh(mean) * self.action_scale + self.action_bias
        action_ret = tanh_action * self.action_scale + self.action_bias
        if return_distribution:
            return action_ret, log_prob, mean_ret, mean, std, ascle, ashft
        else:
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