# import this before torch
from comet_ml import Experiment

import argparse
import gym
import numpy as np
import itertools
import torch
import json
from arrl import ARRL
from pylint.test.functional import return_in_init
from tensorboardX import SummaryWriter
from replay_buffer import ReplayBuffer


parser = argparse.ArgumentParser(description='PyTorch AutoRegressiveFlows-RL Args')
# Once we get Mujoco then we will use this one
parser.add_argument('--env-name', default="HalfCheetah-v2",
                     help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
################ Specific to ARSAC ####################
parser.add_argument('--use_gated_transform', type=bool, default=False, metavar='G',
                    help='Use Inverse Autoregressive Flow')
parser.add_argument('--ignore_scale', type=bool, default=False, metavar='G',
                    help='Causes normal autoregressive flow to only have a shift component')
parser.add_argument('--use_prev_states', type=bool, default=False, metavar='G',
                    help='Determines whether or not to use previous states as well as actions')
parser.add_argument('--action_lookback', type=int, default=5, metavar='G',
                    help='Use phi network to de-correlate time dependence and state by using previous action(s)')
parser.add_argument('--add_state_noise', type=bool, default=False, metavar='G',
                    help='Adds a small amount of Gaussian noise to the state')
parser.add_argument('--add_action_noise', type=bool, default=False, metavar='G',
                    help='Adds a small amount of Gaussian noise to the actions')
#######################################################
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--eval_steps', type=int, default=5000, metavar='N',
                    help='Steps between each evaluation episode')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

with open('models/' + args.env_name + '_parser_args_' + str(args.action_lookback) + '.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Environment
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
action_space_size = env.action_space.sample().shape[0]
state_space_size = env.reset().shape[0]

# Agent
agent = ARRL(env.observation_space.shape[0], env.action_space, args)

# Comet logging
experiment = Experiment(api_key="tHDbEydFQGW7F1MWmIKlEvrly",
                        project_name="arsac_test", workspace="aguerra")
experiment.log_parameters(args.__dict__)

# Memory
memory = ReplayBuffer(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0
stop_training = False

with experiment.train():
    for i_episode in itertools.count(1):

        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        lookback = args.action_lookback
        # Reset the previous action, as our program factors this into account when taking future actions
        if lookback > 0:
            prev_actions = np.zeros(action_space_size * lookback)
            prev_states = np.zeros(state_space_size * lookback)
        else:
            prev_actions = None
            prev_states = None

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:

                # Sample action from policy, adding noise to state if we want to
                state_noise = np.random.normal(0, 0.1, state_space_size) if args.add_state_noise else 0
                action = agent.select_action(state + state_noise, prev_states, prev_actions)

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    # Log to Comet.ml
                    experiment.log_metric("Critic_1_Loss", critic_1_loss, step=updates)
                    experiment.log_metric("Critic_2_Loss", critic_2_loss, step=updates)
                    experiment.log_metric("Policy_Loss", policy_loss, step=updates)
                    experiment.log_metric("Entropy_Loss", ent_loss, step=updates)

                    updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            # Append transition to memory
            memory.push(prev_states, prev_actions, state, action, reward, next_state, mask)

            if lookback > 0:
                prev_actions = np.concatenate((prev_actions[action_space_size:], action))
                prev_states = np.concatenate((prev_states[state_space_size:], state))

            state = next_state

            # # Do an eval episode every <eval_steps> steps
            if total_numsteps % args.eval_steps == 0: #and total_numsteps >= args.start_steps:
                # Save the environment state of the run we were just doing.
                temp_state = env.sim.get_state()
                avg_reward_eval = 0.
                episodes_eval = 1 # Only do 1 episode for each evaluation. If you do more will screw up logging.

                # This dictionary will be useful for logging to Comet.
                episode_eval_dict = {'state': [], 'action': [], 'reward': [], 'qpos': [], 'qvel': [],
                                     'base_mean': [], 'base_std': [], 'adj_scale': [], 'adj_shift': []}

                for _ in range(episodes_eval):
                    state_eval = env.reset()
                    if lookback > 0:
                        prev_actions_eval = np.zeros(action_space_size * lookback)
                        prev_states_eval = np.zeros(state_space_size * lookback)
                    else:
                        prev_actions_eval = None
                        prev_states_eval = None
                    episode_reward_eval = 0
                    done_eval = False
                    while not done_eval:
                        # Sample action from policy, this time taking the mean action
                        action_eval, bmean, bstd, ascle, ashft = \
                            agent.select_action(state_eval, prev_states_eval, prev_actions_eval, eval=True, return_distribution=True)

                        if lookback > 0:
                            prev_actions_eval = np.concatenate((prev_actions_eval[action_space_size:], action_eval))
                            prev_states_eval = np.concatenate((prev_states_eval[state_space_size:], state_eval))
                        # Before we step in the environment, save the Mujoco state (qpos and qvel)
                        episode_eval_dict['qpos'].append(env.sim.get_state()[1].tolist()) # qpos
                        episode_eval_dict['qvel'].append(env.sim.get_state()[2].tolist()) # qvel
                        # Now we step forward in the environment by taking our action
                        next_state_eval, reward_eval, done_eval, _ = env.step(action_eval)
                        # We have completed an evaluation step, now log it to the dictionary
                        episode_eval_dict['state'].append(state_eval.tolist())
                        episode_eval_dict['action'].append(action_eval.tolist())
                        episode_eval_dict['reward'].append(reward_eval.tolist())
                        # Also add stats about the output of our neural networks:
                        episode_eval_dict['base_mean'].append(bmean.tolist())
                        episode_eval_dict['base_std'].append(bstd.tolist())
                        episode_eval_dict['adj_scale'].append(ascle.tolist())
                        episode_eval_dict['adj_shift'].append(ashft.tolist())

                        # Move to the next state
                        state_eval = next_state_eval

                        episode_reward_eval += reward_eval
                        if done_eval:
                            break

                    avg_reward_eval += episode_reward_eval
                avg_reward_eval /= episodes_eval

                # Log the eval reward to Comet-ML
                experiment.log_metric("Avg. Episode_Reward", avg_reward_eval, step=int(total_numsteps / args.eval_steps))

                # Log the episode reward to Comet.ml
                for item_str in episode_eval_dict.keys():
                    item = episode_eval_dict[item_str]
                    json_str = json.dumps(item, separators=(",", ":"), ensure_ascii=False).encode('utf8')
                    item_name = 'episode_step_' + str(total_numsteps) + "_" + item_str
                    experiment.log_asset_data(item, name=item_name, step=int(total_numsteps / args.eval_steps))

                #print("----------------------------------------")
                #print("Test Episodes: {}, Avg. Reward: {}".format(episodes_eval, round(avg_reward_eval, 2)))
                #print("----------------------------------------")

                # Now we are done evaluating. Before we leave, we have to set the state properly.
                env.sim.set_state(temp_state)

            if total_numsteps >= args.num_steps:
                stop_training = True
                break

        # Log to comet.ml
        experiment.log_metric("Epsiode_Reward", episode_reward, step=i_episode)
        # Log to console
        #print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
        #                                                                              episode_steps,
        #                                                                              round(episode_reward, 2)))

        if stop_training:
            break


# Save the final model before finishing program
agent.save_model(args.env_name,
                         actor_path="models/" + args.env_name + "_actor_" + str(lookback) + ".model",
                         critic_path="models/" + args.env_name + "_critic_" + str(lookback) + ".model")

env.close()