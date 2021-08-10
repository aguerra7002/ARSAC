import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform

LOG_SIG_MAX = 10.0
LOG_SIG_MIN = -10.0
epsilon = 1e-6

DELTA_MAX = 10.0
DELTA_MIN = -10.0

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


class ConvQNetwork(nn.Module):
    def __init__(self, num_channels, state_lookback, num_actions, action_lookback, hidden_dim):
        super(ConvQNetwork, self).__init__()

        self.state_lookback = state_lookback
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.action_lookback = action_lookback

        # Q1 architecture
        self.conv1 = nn.Conv2d(num_channels * (state_lookback + 1), 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        # The 64 represents the batch size
        self.linear1 = nn.Linear(4 * 256 + num_actions * (action_lookback + 1), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.conv5 = nn.Conv2d(num_channels * (state_lookback + 1), 32, 4, stride=2)
        self.conv6 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv7 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv8 = nn.Conv2d(128, 256, 4, stride=2)
        # The 64 represents the batch size
        self.linear4 = nn.Linear(4 * 256 + (num_actions * (action_lookback + 1)), hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, prev_states, prev_actions):
        if self.state_lookback > 0:
            s1 = torch.cat((prev_states[:, -self.num_channels * self.state_lookback:], state), 1)
        else:
            s1 = state
        if self.action_lookback > 0:
            a1 = torch.cat((prev_actions[:, -self.num_actions * self.action_lookback:], action), 1)
        else:
            a1 = action

        x1 = F.relu(self.conv1(s1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = x1.view(x1.size(0), -1)
        x1 = torch.cat([x1, a1], 1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.conv1(s1))
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))
        x2 = x2.view(x2.size(0), -1)
        x2 = torch.cat([x2, a1], 1)
        x2 = F.relu(self.linear4(x2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class QNetwork(nn.Module):
    def __init__(self, num_inputs, state_lookback, num_actions, action_lookback, hidden_dim):
        super(QNetwork, self).__init__()

        self.state_lookback = state_lookback
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.action_lookback = action_lookback

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs * (state_lookback + 1) + num_actions * (action_lookback + 1), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs * (state_lookback + 1) + num_actions * (action_lookback + 1), hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, previous_states, previous_actions):

        if self.state_lookback > 0:
            s1 = torch.cat((previous_states[:, -self.num_inputs * self.state_lookback:], state), 1)
        else:
            s1 = state
        if self.action_lookback > 0:
            a1 = torch.cat((previous_actions[:, -self.num_actions * self.action_lookback:], action), 1)
        else:
            a1 = action

        xu = torch.cat([s1, a1], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None,
                 action_lookback=0, state_lookback=False, use_gated_transform=False, ignore_scale=False,
                 hidden_dim_base=256, pixel_based=False):
        super(GaussianPolicy, self).__init__()
        # Specifying the Theta Network (will map states to some latent space equal in dimension to action space)
        self.hidden_dim_base = hidden_dim_base
        self.pixel_based = pixel_based
        if not pixel_based:
            self.linear_theta_1 = nn.Linear(num_inputs * (state_lookback + 1), hidden_dim_base)
            if hidden_dim_base == 256:
                self.linear_theta_2 = nn.Linear(hidden_dim_base, hidden_dim_base)
            self.mean_linear_theta = nn.Linear(hidden_dim_base, num_actions)
            self.log_std_linear_theta = nn.Linear(hidden_dim_base, num_actions)
        else:
            self.conv1 = nn.Conv2d(3 * (state_lookback + 1), 32, 4, stride=2)  # image has 3 channels
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
            self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
            # TODO: Add in some way to change the size of these layers via 'hidden_dim_base' param
            self.mean_linear_theta = nn.Linear(4 * 256, num_actions)
            self.log_std_linear_theta = nn.Linear(4 * 256, num_actions)

        self.state_space_size = num_inputs
        self.action_space_size = num_actions
        # How far back we look with the phi network
        self.action_lookback = action_lookback
        # Do we use prev actions and states? Or just prev actions
        self.state_lookback = state_lookback
        # Do we use inverse autoregressive transform
        self.use_gated_transform = use_gated_transform
        # Do we use a scaling factor of 1?
        self.ignore_scale = ignore_scale

        if action_lookback > 0:
            self.linear_phi_1 = nn.Linear(num_actions * action_lookback, hidden_dim)
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



    # Tracks the l1 or l2 loss
    def get_reg_loss(self, lambda_reg=0.0, use_l2_reg=False):
        # Don't waste time if no regularization
        if lambda_reg == 0.0:
            return 0

        reg_loss = 0
        norm_type = 'fro' if use_l2_reg else 'nuc'
        if not self.pixel_based:
            for name, param in self.linear_theta_1.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            for name, param in self.mean_linear_theta.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            for name, param in self.log_std_linear_theta.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            if self.hidden_dim_base == 256:
                for name, param in self.linear_theta_2.named_parameters():
                    if 'weight' in name:
                        reg_loss += torch.norm(param, p=norm_type)
        else:
            pass  # TODO; implement regularization for CNN layers
        return reg_loss * lambda_reg

    def forward_theta(self, state, previous_states):
        if self.state_lookback > 0:
            inp = torch.cat((previous_states[:, -self.state_space_size * self.state_lookback:], state), 1)
        else:
            inp = state

        if not self.pixel_based:
            x = F.relu(self.linear_theta_1(inp))
            if self.hidden_dim_base == 256:
                x = F.relu(self.linear_theta_2(x))
        else:
            x = F.relu(self.conv1(inp))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.view(x.size(0), -1)
        mean = self.mean_linear_theta(x)
        log_std = self.log_std_linear_theta(x)
        # Soft learning:
        log_std = LOG_SIG_MAX - F.softplus(LOG_SIG_MAX - log_std)
        log_std = LOG_SIG_MIN + F.softplus(log_std - LOG_SIG_MIN)
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # <- what we had before
        return mean, log_std

    def forward_phi(self, prev_actions):

        inp = prev_actions[:, -self.action_space_size * self.action_lookback:]
        x = F.relu(self.linear_phi_1(inp))
        x = F.relu(self.linear_phi_2(x))

        if self.use_gated_transform:
            # Using inverse ar flows
            sigma = torch.sigmoid(self.log_scale_linear_phi(x))
            m = self.shift_linear_phi(x)
            # m = torch.clamp(m) # TODO: See if we need to clamp?
            return m, sigma
        else:
            shift = self.shift_linear_phi(x)
            shift = torch.clamp(shift, min=-5.0, max=5.0)
            log_scale = self.log_scale_linear_phi(x)
            # Soft learning
            log_scale = LOG_SIG_MAX - F.softplus(LOG_SIG_MAX - log_scale)
            log_scale = LOG_SIG_MIN + F.softplus(log_scale - LOG_SIG_MIN)
            # log_scale = torch.clamp(log_scale, min=LOG_SIG_MIN, max=LOG_SIG_MAX) <- What we had before.
            return shift, log_scale

    def sample(self, state, prev_states, prev_actions, return_distribution=False, random_base=False):
        # First pass the state through the state network
        mean1, log_std = self.forward_theta(state, prev_states)
        std = log_std.exp()

        if random_base:
            mean1 = torch.zeros(mean1.shape)
            std = torch.ones(std.shape)

        # Sample from the base distribution first.
        base_dist = Normal(mean1, std)
        base_action = base_dist.rsample()

        # Base distribution
        log_prob = base_dist.log_prob(base_action)

        if self.action_lookback > 0:
            m, sigma = self.forward_phi(prev_actions)
            if self.use_gated_transform:
                # Inverse ar transform
                action = sigma * base_action + (1 - sigma) * m
                log_prob -= sigma.log()  # Convert back to log so prb is additive
                # For logging
                ascle = sigma
                ashft = m
            else:
                # If we are here, we are doing the standard autoregressive transform
                # Now we adjust the mean and std based on the outputs from phi_network
                if self.ignore_scale:
                    # If we only want to incorporate the shift execute this code
                    action = base_action + m
                    mean = mean1 + m
                    # For logging
                    ascle = torch.ones(base_action.shape)
                    ashft = m
                else:
                    action = base_action * sigma.exp() + m
                    # Also want to adjust the mean, so that evaluation mode also works
                    mean = mean1 * sigma.exp() + m
                    # Affine Transform Scaling
                    log_prob -= sigma
                    # For logging
                    ascle = sigma.exp()
                    ashft = m

        else:
            action = base_action
            mean = mean1
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
            return action_ret, log_prob, mean_ret, mean1, std, ascle, ashft
        else:
            return action_ret, log_prob, mean_ret

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)




"""
Here is the experimental class for using previous actions to estimate a prior on actions.
"""

class GaussianPolicy2(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, action_lookback=0, hidden_dim_base=256):
        super(GaussianPolicy2, self).__init__()
        # Specifying the Theta Network (will map states to some latent space equal in dimension to action space)
        self.hidden_dim_base = hidden_dim_base
        # This will take in the state along with the action prior
        self.linear_theta_1 = nn.Linear(num_inputs + 2 * num_actions, hidden_dim_base)
        self.layer_norm_theta_1 = nn.LayerNorm(hidden_dim_base)
        if hidden_dim_base == 256:
            self.linear_theta_2 = nn.Linear(hidden_dim_base, hidden_dim_base)
            self.layer_norm_theta_2 = nn.LayerNorm(hidden_dim_base)
        self.delta_mean_linear_theta = nn.Linear(hidden_dim_base, num_actions)
        self.sigma_mean_linear_theta = nn.Linear(hidden_dim_base, num_actions)
        self.delta_std_linear_theta = nn.Linear(hidden_dim_base, num_actions)
        self.sigma_std_linear_theta = nn.Linear(hidden_dim_base, num_actions)

        self.state_space_size = num_inputs
        self.action_space_size = num_actions
        # How far back we look with the phi network
        self.action_lookback = action_lookback
        # This will be our prior network
        if action_lookback == 0:
            self.linear_phi_1 = nn.Linear(num_inputs, hidden_dim) # SAC with state prior.
        else:
            self.linear_phi_1 = nn.Linear(num_actions * action_lookback, hidden_dim)
        self.layer_norm_phi_1 = nn.LayerNorm(hidden_dim)
        self.linear_phi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm_phi_2 = nn.LayerNorm(hidden_dim)
        self.log_std_linear_phi = nn.Linear(hidden_dim, num_actions)
        self.mean_linear_phi = nn.Linear(hidden_dim, num_actions)


        self.apply(weights_init_)
        # Will make the initial adjustment close to 0 (to avoid large KLs)
        # First the prior network
        nn.init.constant_(self.mean_linear_phi.weight, 0)
        nn.init.constant_(self.mean_linear_phi.bias, 0)
        nn.init.constant_(self.log_std_linear_phi.weight, 0)
        nn.init.constant_(self.log_std_linear_phi.bias, 3) # Initialize the prior to have high variance
        # Then the state action network
        nn.init.constant_(self.delta_mean_linear_theta.weight, 0)
        nn.init.constant_(self.delta_std_linear_theta.weight, 0)
        nn.init.constant_(self.delta_mean_linear_theta.bias, 0)
        nn.init.constant_(self.delta_std_linear_theta.bias, 3) # Initialize the state-action mapping to have high variance.
        nn.init.constant_(self.sigma_mean_linear_theta.weight, 0)
        nn.init.constant_(self.sigma_std_linear_theta.weight, 0)
        nn.init.constant_(self.sigma_mean_linear_theta.bias, -4) # <- initial policy will primarily use the state-action mapping
        nn.init.constant_(self.sigma_std_linear_theta.bias, 0)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    # Tracks the l1 or l2 loss
    def get_reg_loss(self, lambda_reg=0.0, use_l2_reg=False):
        # Don't waste time if no regularization
        if lambda_reg == 0.0:
            return 0

        reg_loss = 0
        norm_type = 'fro' if use_l2_reg else 'nuc'
        if not self.pixel_based:
            for name, param in self.linear_theta_1.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            for name, param in self.delta_mean_linear_theta.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            for name, param in self.sigma_mean_linear_theta.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            for name, param in self.delta_std_linear_theta.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            for name, param in self.sigma_std_linear_theta.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, p=norm_type)
            if self.hidden_dim_base == 256:
                for name, param in self.linear_theta_2.named_parameters():
                    if 'weight' in name:
                        reg_loss += torch.norm(param, p=norm_type)
        return reg_loss * lambda_reg

    def forward_theta(self, state, prior_mean, prior_log_std):

        inp = torch.cat((state, prior_mean, prior_log_std), 1)

        x = F.relu(self.layer_norm_theta_1(self.linear_theta_1(inp)))
        if self.hidden_dim_base == 256:
            x = F.relu(self.layer_norm_theta_2(self.linear_theta_2(x)))
        delta_mean = self.delta_mean_linear_theta(x)
        sigma_mean = self.sigma_mean_linear_theta(x)
        delta_log_std = self.delta_std_linear_theta(x)
        sigma_std = self.sigma_std_linear_theta(x)

        delta_log_std = DELTA_MAX - F.softplus(DELTA_MAX - delta_log_std)
        delta_log_std = DELTA_MIN + F.softplus(delta_log_std - DELTA_MIN)

        # Scale between 0 and 2 (would be 0 and 1 for the gate)
        sigma_mean = torch.sigmoid(sigma_mean) * 1
        sigma_std = torch.sigmoid(sigma_std) * 1

        return delta_mean, sigma_mean, delta_log_std, sigma_std

    def forward_phi(self, prev_actions):

        inp = prev_actions[:, -self.action_space_size * self.action_lookback:]
        x = F.relu(self.layer_norm_phi_1(self.linear_phi_1(inp)))
        x = F.relu(self.layer_norm_phi_2(self.linear_phi_2(x)))

        mean = self.mean_linear_phi(x)
        log_std = self.log_std_linear_phi(x)
        # Soft learning
        log_std = LOG_SIG_MAX - F.softplus(LOG_SIG_MAX - log_std)
        log_std = LOG_SIG_MIN + F.softplus(log_std - LOG_SIG_MIN)
        # log_scale = torch.clamp(log_scale, min=LOG_SIG_MIN, max=LOG_SIG_MAX) <- What we had before.
        return mean, log_std

    def sample(self, state, prev_states, prev_actions, return_distribution=False, random_base=False, uniform_weight=0):
        # Generate our prior
        if self.action_lookback == 0: # SAC with state-prior for baseline
            mean, log_std = self.forward_phi(state)
        else:
            mean, log_std = self.forward_phi(prev_actions)
        prior_dist = Normal(mean, log_std.exp())
        # sampled_action = prior_dist.rsample()

        # Detach the mean/std so we don't
        delta_mean, sigma_mean, delta_log_std, sigma_std = self.forward_theta(state, mean.detach(), log_std.detach())

        if torch.isnan(delta_mean).any():
            print("Delta Mean NaN")
        if torch.isnan(delta_log_std).any():
            print("Delta STD NaN")
        if torch.isnan(sigma_mean).any():
            print("Sigma Mean NaN")
        if torch.isnan(sigma_std).any():
            print("Sigma STD NaN")

        # Below we give three methods of parameterizing the policy

        # Method #1 (Gated)
        mean_adj = sigma_mean * mean.detach() + (1 - sigma_mean) * delta_mean
        log_std_adj = sigma_std * log_std.detach() + (1-sigma_std) * delta_log_std

        # Method #2 (Scale + Shift)
        #mean_adj = sigma_mean * mean.detach() + delta_mean
        # action_adj = sigma_mean * sampled_action.detach() + delta_mean
        #log_std_adj = sigma_std * log_std.detach() + delta_log_std

        std_adj = log_std_adj.exp()

        # sanity check testing
        # uniform_mix = Uniform(-(self.action_scale + self.action_bias), self.action_scale + self.action_bias)
        # if torch.isnan(mean_adj).any() or torch.isnan(log_std_adj).any():
        #     print("NaN's in Mean/Log STD (adjusted)")
        # if torch.isinf(mean_adj).any() or torch.isinf(log_std_adj).any():
        #     print("Infs in Mean/ Log STD (adjusted)")

        # Normalizing flow interpretation (doesn't work for some reason)
        # log_prob_base = prior_dist.log_prob(sampled_action).detach() - torch.log(sigma_mean)
        # log_prob_prior = prior_dist.log_prob(action_adj)

        # Inference Interpretation
        base_dist = Normal(mean_adj, std_adj)
        action_adj = base_dist.rsample()
        log_prob_prior = prior_dist.log_prob(action_adj.detach())
        log_prob_base = base_dist.log_prob(action_adj)

        # Tanh the action
        tanh_action = torch.tanh(action_adj)
        mean_ret = torch.tanh(mean_adj) * self.action_scale + self.action_bias
        action_ret = tanh_action * self.action_scale + self.action_bias

        #log_prob_uniform = uniform_mix.log_prob(action_ret)

        # Enforcing Action Bound
        log_prob_base -= torch.log(self.action_scale * (1 - tanh_action.pow(2)) + epsilon)
        log_prob_base = log_prob_base.sum(1, keepdim=True)

        log_prob_prior -= torch.log(self.action_scale * (1 - tanh_action.detach().pow(2)) + epsilon) # !!!!!
        log_prob_prior = log_prob_prior.sum(1, keepdim=True)
        # No need to worry about transforming the uniform dist since we are already in the bounded action space
        #log_prob_uniform = log_prob_uniform.sum(1, keepdim=True)
        # Here we introduce the uniform mix to hopefully lessen the magnitude of the KL
        #log_prob_prior = torch.log(log_prob_prior.exp() * (1 - uniform_weight) + log_prob_uniform.exp() * uniform_weight + epsilon)
        # Split the KL into two parts, one for policy_loss and one for ent_loss so gradients don't interfere with each other
        kl_div = (log_prob_base - log_prob_prior.detach(), log_prob_base.detach() - log_prob_prior)
        # kl_div2 = log_prob_base - log_prob_prior
        if return_distribution:
            return action_ret, kl_div, mean_ret.detach(), mean, \
                   log_std.exp(), sigma_mean.detach(), delta_log_std.exp().detach() #delta_mean.detach()
        else:
            return action_ret, kl_div, mean_ret.detach()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy2, self).to(device)
