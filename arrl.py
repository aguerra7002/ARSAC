import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from utils import soft_update, hard_update
from pixelstate import PixelState
from model import GaussianPolicy, QNetwork, ConvQNetwork, GaussianPolicy2

# If we want to Time Profile
PROFILING = False
if PROFILING:
    import time

# How often we update the copy of the policy: TODO: Make this a command-line argument
UPDATE_POLICY_STEP = 1000


class ARRL(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.state_space_size = num_inputs
        self.action_space_size = action_space.shape[0]

        self.action_lookback_actor = args.action_lookback_actor
        self.action_lookback_critic = args.action_lookback_critic
        self.state_lookback_actor = args.state_lookback_actor
        self.state_lookback_critic = args.state_lookback_critic
        self.ignore_scale = args.ignore_scale
        self.use_gated_transform = args.use_gated_transform
        self.lambda_reg = args.lambda_reg
        self.use_l2_reg = args.use_l2_reg

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.restrict_policy_deviation = args.restrict_policy_deviation

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.pixel_based = args.pixel_based

        if args.pixel_based:
            self.state_getter = PixelState(1, args.env_name, args.task_name, args.resolution, num_inputs)
            self.critic = ConvQNetwork(3, self.state_lookback_critic,  # 3 channels
                                       action_space.shape[0], self.action_lookback_critic,
                                       args.hidden_size).to(device=self.device)
        else:
            self.critic = QNetwork(num_inputs, self.state_lookback_critic,
                                   action_space.shape[0], self.action_lookback_critic,
                                   args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        if args.pixel_based:
            self.critic_target = ConvQNetwork(3, self.state_lookback_critic,
                                              action_space.shape[0], self.action_lookback_critic,
                                              args.hidden_size).to(device=self.device)
        else:
            self.critic_target = QNetwork(num_inputs, self.state_lookback_critic,
                                          action_space.shape[0], self.action_lookback_critic,
                                          args.hidden_size).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning:
                self.target_entropy = torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space,
                                         self.action_lookback_actor, self.state_lookback_actor,
                                         self.use_gated_transform,
                                         self.ignore_scale, args.hidden_dim_base, args.pixel_based).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        elif self.policy_type == "Gaussian2":
            if self.automatic_entropy_tuning:
                self.target_entropy = torch.Tensor((args.kl_constraint,)).item()
                # self.target_entropy = (1. + float(np.log(2))) * torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                # For adjusting target kl with old policy
                self.log_beta = torch.zeros(1, requires_grad=True, device=self.device)
                self.beta_optim = Adam([self.log_alpha], lr=args.lr)  # Initialize same as alpha bc why not?
                # For adjusting KL versus expert


            self.policy = GaussianPolicy2(num_inputs, action_space.shape[0], args.hidden_size, action_space,
                                          self.action_lookback_actor, args.hidden_dim_base).to(self.device)
            # This is what we will update every 500 steps in order to not allow our policy to change too much from this.
            if self.restrict_policy_deviation > 0:
                self.policy_old = copy.deepcopy(self.policy)
            # Make sure to be able to specify a different learning rate for the autoregressive network
            all_params = set(self.policy.parameters())
            ar_params = []
            for name, p in self.policy.named_parameters():
                if "phi" in name:
                    ar_params.append(p)
            base_params = list(all_params - set(ar_params))

            self.policy_optim = Adam([{"params": base_params},
                                      {"params": ar_params, "lr": args.lr_ar}],
                                     lr=args.lr)

            # Create 2 separate optimizers, one for AR component and one for base.
            # This gives us more flexibility in our optimization.
            self.base_optim = Adam(base_params, lr=args.lr)
            self.ar_optim = Adam(ar_params, lr=args.lr_ar)

            # For transfer loss
            self.transfer_loss = 0.0
        else:
            print("Shouldn't be here")
            exit(0)

    def select_action(self, state, prev_states=None, prev_actions=None, eval=False, return_distribution=False,
                      random_base=False, return_prob=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze_(0)

        if prev_actions is not None:
            prev_actions = torch.FloatTensor(prev_actions).to(self.device).unsqueeze_(0)
        if prev_states is not None:
            prev_states = torch.FloatTensor(prev_states).to(self.device).unsqueeze_(0)

        if not eval:
            if return_distribution:
                # We pick an action based off a gaussian policy to encourage the model to explore
                action, log_prob, _, bmean, bstd, (ascle, tmp1), (tmp2, ashft) = self.policy.sample(state, prev_states,
                                                                                                    prev_actions,
                                                                                                    return_distribution=True,
                                                                                                    random_base=random_base)
            else:
                # We pick an action based off a gaussian policy to encourage the model to explore
                action, log_prob, _ = self.policy.sample(state, prev_states, prev_actions, random_base=random_base)
        else:
            if return_distribution:
                _, log_prob, action, bmean, bstd, (ascle, tmp1), (tmp2, ashft) = self.policy.sample(state, prev_states, prev_actions,
                                                                                                    return_distribution=True,
                                                                                                    random_base=random_base)
            else:
                _, log_prob, action = self.policy.sample(state, prev_states, prev_actions, random_base=random_base)

        if return_prob and return_distribution:
            return self.to_numpy(action), self.to_numpy(bmean), self.to_numpy(bstd), self.to_numpy(
                ascle), self.to_numpy(ashft), log_prob
        elif return_distribution:
            return self.to_numpy(action), self.to_numpy(bmean), self.to_numpy(bstd), self.to_numpy(
                ascle), self.to_numpy(ashft)
        else:
            return action.detach().cpu().numpy()[0]

    def to_numpy(self, arr):
        return arr.detach().cpu().numpy()[0]

    def require_flow_grad(self, requires_grad):
        for name, param in self.policy.named_parameters():
            if "phi" in name:
                param.requires_grad = requires_grad

    def update_parameters(self, memory, batch_size, updates, restrict_base_output=0.0, step=None, freeze_flow=False,
                          expert=None):
        # Sample a batch from memory
        if PROFILING:
            print("update parameters")
            prev_time = time.time()

        prev_state_batch, prev_action_batch, state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            memory.sample(batch_size=batch_size)
        if self.pixel_based:
            state_batch = self.state_getter.get_pixel_state(state_batch)
            next_state_batch = self.state_getter.get_pixel_state(next_state_batch)
            if None not in prev_state_batch:
                prev_state_batch = self.state_getter.get_pixel_state(prev_state_batch)

        if PROFILING:
            print("\tBatch Sample:", time.time() - prev_time)
            prev_time = time.time()

        prev_next_state_batch = None
        prev_next_action_batch = None

        if None not in prev_state_batch:
            # we need to put together prev_next_state_batch for feeding to the actor network later.
            if self.state_lookback_actor > 0 or self.state_lookback_critic > 0:
                cutoff = 3 if self.pixel_based else self.state_space_size
                prev_next_state_batch = np.concatenate((prev_state_batch[:, cutoff:], state_batch), axis=1)
                prev_next_state_batch = torch.cuda.FloatTensor(prev_next_state_batch)
            prev_state_batch = torch.cuda.FloatTensor(prev_state_batch)

        if None not in prev_action_batch:
            # Same as with states, need to compute the prev_next_action_batch
            prev_next_action_batch = np.concatenate((prev_action_batch[:, self.action_space_size:], action_batch),
                                                    axis=1)
            prev_next_action_batch = torch.cuda.FloatTensor(prev_next_action_batch)
            prev_action_batch = torch.cuda.FloatTensor(prev_action_batch).to(self.device)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze_(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze_(1)

        if PROFILING:
            print("\tSetup:", time.time() - prev_time)
            prev_time = time.time()
        # Anneal the uniform weight parameter
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = \
                self.policy.sample(next_state_batch, prev_next_state_batch, prev_next_action_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action,
                                                                  prev_next_state_batch, prev_next_action_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi[0]
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

            if PROFILING:
                print("\tAct and Critic:", time.time() - prev_time)
                prev_time = time.time()

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch, prev_state_batch, prev_action_batch)

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q2(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        sampled_action_batch, log_pi, _, policy_mean, policy_std, policy_dist = \
            self.policy.sample(state_batch, prev_state_batch, prev_action_batch,
                               return_policy=True)
        prior_current = Normal(policy_mean, policy_std)
        # sampled_action_batch should be detached from the prior.

        qf1_pi, qf2_pi = self.critic(state_batch, sampled_action_batch, prev_state_batch, prev_action_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Want to use this to update our state action mapping
        # policy_loss = -min_qf_pi.mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = self.alpha * log_pi[0].mean() - min_qf_pi.mean()

        # Want to use this as the loss for the AR component of the policy
        ent_loss = log_pi[1].mean()

        # Here is where we adjust the policy loss to ensure we do not deviate too much from the policy during training.
        if self.restrict_policy_deviation > 0 and step is not None:
            # Here we do the same thing with the old policy so that we can adjust the lost accordingly
            _, _, _, policy_mean_old, policy_std_old, _, _ = \
                self.policy_old.sample(state_batch, prev_state_batch, prev_action_batch,
                                       return_distribution=True)

            prior_old = Normal(policy_mean_old.detach(), policy_std_old.detach())
            kl_div_from_old = torch.distributions.kl_divergence(prior_current, prior_old).mean()
            ent_loss += kl_div_from_old * self.restrict_policy_deviation

            if self.automatic_entropy_tuning:
                beta_loss = (self.log_beta.exp() * kl_div_from_old.detach()).mean()
                self.beta_optim.zero_grad()
                beta_loss.backward()
                self.beta_optim.step()
                self.restrict_policy_deviation = self.log_beta.exp()[0]

        # Add in regularization to policy here.
        # policy_loss += self.policy.get_reg_loss(lambda_reg=self.lambda_reg, use_l2_reg=self.use_l2_reg)
        #
        # # Add loss if we choose to restrict the output of the network
        # if restrict_base_output > 0.0 and self.policy_type == "Gaussian":
        #     norm_type = 'fro' if self.use_l2_reg else 'nuc'
        #     norms = torch.norm(policy_mean, dim=1, p=norm_type).mean() + torch.norm(policy_std.log(), dim=1, p=norm_type).mean()
        #     policy_loss += norms * restrict_base_output
        #
        # if restrict_base_output > 0.0 and self.policy_type == "Gaussian2":
        #     norm_type = 'fro' if self.use_l2_reg else 'nuc'
        #     # Keep the scale close to 1 and keep the delta close to 0.
        #     norms = torch.norm(torch.log(sigma_mean), dim=1, p=norm_type).mean() + torch.norm(delta_mean, dim=1, p=norm_type).mean()
        #     policy_loss += norms * restrict_base_output

        # Here is where we add our transfer loss
        if self.transfer_loss > 0 and expert is not None:
            prior_expert = expert.policy.generate_prior(prev_action_batch)
            # Method 1. Try enforcing a constraint on the prior
            # expert_prior_kl = torch.distributions.kl_divergence(prior_expert, prior_current).mean()
            # ent_loss += self.transfer_loss * expert_prior_kl # TODO: Get rid of transfer loss parameter, use beta instead.

            expert_policy_kl = torch.distributions.kl_divergence(prior_expert, policy_dist).mean()
            policy_loss += self.transfer_loss * expert_policy_kl
            # Try adding a loss term to the policy as well? idk
            # Ensure that beta is tuned to reduce
            # if self.automatic_entropy_tuning:
            #     beta_loss = (self.log_beta.exp() * expert_kl.detach())
            #     self.beta_optim.zero_grad()
            #     beta_loss.backward()
            #     self.beta_optim.step()

        # Finally we update the policy
        if self.policy_type == "Gaussian":
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        elif self.policy_type == "Gaussian2":

            # Update the AR component based on KL loss (called ent_loss here)
            if not freeze_flow:
                self.ar_optim.zero_grad()
                ent_loss.backward(retain_graph=True)
                self.ar_optim.step()

            # Update the state-action network based on standard policy loss
            self.base_optim.zero_grad()
            policy_loss.backward()
            self.base_optim.step()

            # Should print all 0's
            # for name, param in self.policy.named_parameters():
            #     if "phi" in name:
            #         print(name, param.grad)

        if PROFILING:
            print("\tCritic Loss:", time.time() - prev_time)
            prev_time = time.time()
        if self.automatic_entropy_tuning:
            # print(torch.min(log_pi).item(), torch.max(log_pi).item(), torch.mean(log_pi).item())
            alpha_loss = (self.log_alpha.exp() * (log_pi[0] - self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if PROFILING:
            print("\tBackprop:", time.time() - prev_time)  # TODO: Remove later
            prev_time = time.time()

        # Final Step: Every 500 steps we reset the "old" policy
        if step is not None and step % UPDATE_POLICY_STEP == 0 and self.restrict_policy_deviation > 0:
            self.policy_old = copy.deepcopy(self.policy)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), ent_loss.item(), alpha_tlogs.item()

    def set_transfer_loss(self, tl):
        self.transfer_loss = tl

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None,
                   ar_optim_path=None, base_optim_path=None, policy_optim_path=None):

        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)

        # Save the optimizers
        if ar_optim_path is not None and self.policy_type == "Gaussian2":
            torch.save(self.ar_optim.state_dict(), ar_optim_path)
        if base_optim_path is not None and self.policy_type == "Gaussian2":
            torch.save(self.base_optim.state_dict(), base_optim_path)
        if policy_optim_path is not None and self.policy_type == "Gaussian":
            torch.save(self.policy_optim.state_dict(), policy_optim_path)

        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path, flow_only=False, base_only=False,
                   ar_optim_path=None, base_optim_path=None, policy_optim_path=None):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            loaded_dict = torch.load(actor_path)
            if flow_only or base_only:
                # Only load the flow network
                for key in self.policy.state_dict().keys():
                    # We can identify a component of the flow network bc they contain the string "phi"
                    if "phi" in key and flow_only:
                        if key in loaded_dict.keys():
                            self.policy.state_dict()[key] = loaded_dict[key]
                            print("Transfer", key)
                        else:
                            print("Uh-oh: Key", key, "not found in the loaded dict.")
                    if "theta" in key and base_only:
                        if key in loaded_dict.keys():
                            self.policy.state_dict()[key] = loaded_dict[key]
                            print("Transfer", key)
                        else:
                            print("Uh-oh: Key", key, "not found in the loaded dict.")
            else:
                self.policy.load_state_dict(loaded_dict)

        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

        # Load the optimizers as well
        if ar_optim_path is not None and self.policy_type == "Gaussian2":
            self.ar_optim.load_state_dict(torch.load(ar_optim_path))
        if base_optim_path is not None and self.policy_type == "Gaussian2":
            self.base_optim.load_state_dict(torch.load(base_optim_path))
        if policy_optim_path is not None and self.policy_type == "Gaussian":
            self.policy_optim.load_state_dict(torch.load(policy_optim_path))
