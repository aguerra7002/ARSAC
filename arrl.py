import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class ARRL(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.action_lookback = args.action_lookback
        self.ignore_scale = args.ignore_scale
        self.use_prev_states = args.use_prev_states
        self.use_gated_transform = args.use_gated_transform

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space,
                                        self.action_lookback, self.use_prev_states, self.use_gated_transform,
                                        self.ignore_scale).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def select_action(self, state, prev_states=None, prev_actions=None, eval=False, return_distribution=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if prev_actions is not None:
            prev_actions = torch.FloatTensor(prev_actions).to(self.device).unsqueeze(0)
        if prev_states is not None:
            prev_states = torch.FloatTensor(prev_states).to(self.device).unsqueeze(0)

        if not eval:
            if return_distribution:
                # We pick an action based off a gaussian policy to encourage the model to explore
                action, _, _, bmean, bstd, ascle, ashft = self.policy.sample(state, prev_states, prev_actions,
                                                                             return_distribution=True)
            else:
                # We pick an action based off a gaussian policy to encourage the model to explore
                action, _, _ = self.policy.sample(state, prev_states, prev_actions)
        else:
            if return_distribution:
                _, _, action, bmean, bstd, ascle, ashft = self.policy.sample(state, prev_states, prev_actions,
                                                                             return_distribution=True)
            else:
                _, _, action = self.policy.sample(state, prev_states, prev_actions)

        if return_distribution:
            return self.to_numpy(action), self.to_numpy(bmean), self.to_numpy(bstd), self.to_numpy(ascle), self.to_numpy(ashft)
        else:
            return action.detach().cpu().numpy()[0]

    def to_numpy(self, arr):
        return arr.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        prev_state_batch, prev_action_batch, state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            memory.sample(batch_size=batch_size)

        prev_next_state_batch = None
        prev_next_action_batch = None

        if None not in prev_state_batch:
            # we need to put together prev_next_state_batch for feeding to the actor network later.
            state_space_size = int(prev_state_batch.shape[1] / self.action_lookback)
            prev_next_state_batch = np.concatenate((prev_state_batch[:, state_space_size:], state_batch), axis=1)
            prev_next_state_batch = torch.FloatTensor(prev_next_state_batch).to(self.device)
            prev_state_batch = torch.FloatTensor(prev_state_batch).to(self.device)

        if None not in prev_action_batch:
            # Same as with states, need to compute the prev_next_action_batch
            action_space_size = int(prev_action_batch.shape[1] / self.action_lookback)
            prev_next_action_batch = np.concatenate((prev_action_batch[:, action_space_size:], action_batch), axis=1)
            prev_next_action_batch = torch.FloatTensor(prev_next_action_batch).to(self.device)
            prev_action_batch = torch.FloatTensor(prev_action_batch).to(self.device)


        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # Old (wrong)
            # next_state_action, next_state_log_pi, _ = \
            #     self.policy.sample(next_state_batch, prev_state_batch, prev_action_batch)
            # New:
            next_state_action, next_state_log_pi, _ = \
                self.policy.sample(next_state_batch, prev_next_state_batch, prev_next_action_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        pi, log_pi, _ = self.policy.sample(state_batch, prev_state_batch, prev_action_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
