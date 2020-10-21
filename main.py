# import this before torch
from comet_ml import Experiment
import os
import argparse
import numpy as np
import itertools
import torch
import json
from arrl import ARRL
from replay_buffer import ReplayBuffer
from pixelstate import PixelState
from env_wrapper import EnvWrapper

parser = argparse.ArgumentParser(description='PyTorch AutoRegressiveFlows-RL Args')
# Once we get Mujoco then we will use this one
parser.add_argument('--env-name', default="walker",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--task-name', default="walk",
                    help='Task name to use in the Deepmind control suite. Leave Blank to use Gym environments')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.03, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
################ Specific to ARSAC ####################
parser.add_argument('--use_gated_transform', type=bool, default=False, metavar='G',
                    help='Use Inverse Autoregressive Flow')
parser.add_argument('--ignore_scale', type=bool, default=False, metavar='G',
                    help='Causes normal autoregressive flow to only have a shift component')
parser.add_argument('--state_lookback_actor', type=int, default=3, metavar='G',
                    help='Determines whether or not to use previous states as well as actions in actor network')
parser.add_argument('--action_lookback_actor', type=int, default=3, metavar='G',
                    help='Use phi network to de-correlate time dependence and state by using previous action(s)')
parser.add_argument('--state_lookback_critic', type=int, default=3, metavar='G',
                    help='Determines how many states we look back when estimating rewards')
parser.add_argument('--action_lookback_critic', type=int, default=3, metavar='G',
                    help='Determines how many actions we look back when estimating rewards')
parser.add_argument('--add_state_noise', type=bool, default=False, metavar='G',
                    help='Adds a small amount of Gaussian noise to the state')
parser.add_argument('--add_action_noise', type=bool, default=False, metavar='G',
                    help='Adds a small amount of Gaussian noise to the actions')
parser.add_argument('--random_base_train', type=bool, default=False, metavar='G',
                    help='Uses a standard Gaussian for the base distribution during training.')
parser.add_argument('--random_base_eval', type=bool, default=False, metavar='G',
                    help='Uses a standard Gaussian for the base distribution during eval episodes.')
parser.add_argument('--hidden_dim_base', type=int, default=32, metavar='G',
                    help='Determines how many hidden units to use for the hidden layer of the state mapping')
parser.add_argument('--lambda_reg', type=float, default=0.0, metavar='G',
                    help='How much regularization to use in base network.')
parser.add_argument('--use_l2_reg', type=bool, default=True, metavar='G',
                    help="Uses l2 regularization on state-action policy if true, otherwise uses l1 regularization")
parser.add_argument('--restrict_base_output', type=float, default=0.0001, metavar='G',
                    help="Restricts output of base network by adding loss based on norm of network output")
parser.add_argument('--position_only', type=bool, default=False, metavar='G',
                    help="Determines whether or not we only use the Mujoco positions versus the entire state. This " +
                         "argument is ignored if pixel_based is True.")
parser.add_argument('--pixel_based', type=bool, default=True, metavar='G',
                    help='Uses a pixel based state as opposed to position/velocity vectors. Do not use with use_prev_states=True')
parser.add_argument('--resolution', type=int, default=64, metavar='G',
                    help='Decides the resolution of the pixel based image. Default is 64x64.')
#######################################################
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=256, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--eval_steps', type=int, default=10000, metavar='N',
                    help='Steps between each evaluation episode')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')
parser.add_argument('--device_id', type=int, default=0, metavar='G',
                    help='Which GPU to run on')
args = parser.parse_args()

if args.device_id is None:
    args.cuda = False
else:
    args.cuda = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.cuda.set_device(0)

# Use our environment wrapper for making the environment
env = EnvWrapper(args.env_name, args.task_name, args.pixel_based, args.resolution)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
action_space_size = env.action_space.sample().shape[0]


state_space_size = env.get_state_space_size(position_only=args.position_only)

# If we want to Time Profile
PROFILING = True
if PROFILING:
    import time

# Agent
agent = ARRL(state_space_size, env.action_space, args)

# Comet logging
experiment = Experiment(api_key="tHDbEydFQGW7F1MWmIKlEvrly",
                        project_name="arsac_test", workspace="aguerra")
experiment.log_parameters(args.__dict__)
json_str = json.dumps(args.__dict__)
experiment.log_asset_data(json_str, name="args")

# This will be what we use to get image states for the main loop. The updating parameter uses a multithreaded version of this class
if args.pixel_based:
    state_getter_main = PixelState(1, args.env_name, args.task_name, args.resolution, state_space_size)

# Memory
memory = ReplayBuffer(args.replay_size)

# Will hold ALL of our results and will be what we log to comet
eval_dicts = []

# Training Loop
total_numsteps = 0
updates = 0
stop_training = False

action_lookback = max(args.action_lookback_actor, args.action_lookback_critic)
state_lookback = max(args.state_lookback_actor, args.state_lookback_critic)

with experiment.train():
    for i_episode in itertools.count(1):

        episode_reward = 0
        episode_steps = 0
        # This will be used to plot the log std/scale parameters
        bstds = []
        ascales = []
        critic_1_losses = []
        critic_2_losses = []
        policy_losses = []
        done = False
        state = env.get_current_state(temp_state=None, position_only=args.position_only)

        # Reset the previous action, as our program factors this into account when taking future actions
        if action_lookback > 0:
            prev_actions = np.zeros(action_space_size * action_lookback)
        else:
            prev_actions = None
        if state_lookback > 0:
            prev_states = np.zeros(state_space_size * state_lookback)
        else:
            prev_states = None

        while not done:
            if PROFILING:
                print("Step #:", total_numsteps)
                prev_time = time.time() # TODO: Remove later
                s_time = prev_time

            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:

                # Sample action from policy, adding noise to state if we want to
                # If pixel based we need to get the image
                if args.pixel_based:
                    state_n = state_getter_main.get_pixel_state(state, batch=False)
                    if state_lookback > 0:
                        prev_states_n = state_getter_main.get_pixel_state(prev_states, batch=False)
                    else:
                        prev_states_n = prev_states
                    action, bmean, bstd, ascle, ashft = agent.select_action(state_n, prev_states_n, prev_actions,
                                                 random_base=args.random_base_train, return_distribution=True)
                    bstds.append(bstd)
                    ascales.append(ascle)
                else:
                    action, bmean, bstd, ascle, ashft = agent.select_action(state, prev_states, prev_actions,
                                                 random_base=args.random_base_train, return_distribution=True)
                    bstds.append(bstd)
                    ascales.append(ascle)
            if PROFILING:
                print("select action:", time.time() - prev_time) # TODO: Remove later
                prev_time = time.time()

            if len(memory) > args.start_steps: #args.batch_size * :
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    critic_1_losses.append(critic_1_loss)
                    critic_2_losses.append(critic_2_loss)
                    policy_losses.append(policy_loss)

                    updates += 1

            if PROFILING:
                prev_time = time.time() # TODO: REmove later

            action_noise = np.random.normal(0, 0.1, action_space_size) if args.add_action_noise else 0
            # Take a step in the environment. Note, we get the next state in the following line in case we only want pos
            tmp_st, reward, done, _ = env.step(action + action_noise)  # Step
            next_state = env.get_current_state(temp_state=tmp_st, position_only=args.position_only)

            if PROFILING:
                print("get state:", time.time() - prev_time)  # TODO: Remove later
                prev_time = time.time()

            # Add state noise if that parameter is true
            #next_state += np.random.normal(0, 0.1, state_space_size) if args.add_state_noise else 0
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            # Append transition to memory
            memory.push(prev_states, prev_actions, state, action, reward, next_state, mask)

            if action_lookback > 0:
                prev_actions = np.concatenate((prev_actions[action_space_size:], action))
            if state_lookback > 0:
                prev_states = np.concatenate((prev_states[state_space_size:], state))

            state = next_state

            if PROFILING:
                print("other stuff:", time.time() - prev_time)  # TODO: Remove later
                prev_time = time.time()

            # # Do an eval episode every <eval_steps> steps
            if total_numsteps % args.eval_steps == 0:  # and total_numsteps >= args.start_steps:
                # Save the environment state of the run we were just doing.
                temp_state = env.get_state_before_eval()
                avg_reward_eval = 0.
                episodes_eval = 1  # Only do 1 episode for each evaluation. If you do more will screw up logging.

                # This dictionary will be useful for logging to Comet.
                episode_eval_dict = {'state': [], 'action': [], 'reward': [], 'qpos': [], 'qvel': [],
                                     'base_mean': [], 'base_std': [], 'adj_scale': [], 'adj_shift': []}

                for _ in range(episodes_eval):
                    state_eval = env.get_current_state(temp_state=None, position_only=args.position_only)

                    if action_lookback > 0:
                        prev_actions_eval = np.zeros(action_space_size * action_lookback)
                    else:
                        prev_actions_eval = None
                    if state_lookback > 0:
                        prev_states_eval = np.zeros(state_space_size * state_lookback)
                    else:
                        prev_states_eval = None

                    episode_reward_eval = 0
                    done_eval = False
                    while not done_eval:
                        # Sample action from policy, this time taking the mean action
                        if args.pixel_based:
                            # If we are doing pixel_based, get the pixel image
                            prev_states_eval_n = state_getter_main.get_pixel_state(prev_states, batch=False)
                            state_eval_n = state_getter_main.get_pixel_state(state, batch=False)
                            action_eval, bmean, bstd, ascle, ashft = \
                                agent.select_action(state_eval_n, prev_states_eval_n, prev_actions_eval,
                                                    eval=True, return_distribution=True,
                                                    random_base=args.random_base_eval)
                        else:
                            action_eval, bmean, bstd, ascle, ashft = \
                                agent.select_action(state_eval, prev_states_eval, prev_actions_eval,
                                                eval=True, return_distribution=True, random_base=args.random_base_eval)

                        if action_lookback > 0:
                            prev_actions_eval = np.concatenate((prev_actions_eval[action_space_size:], action_eval))
                        if state_lookback > 0:
                            prev_states_eval = np.concatenate((prev_states_eval[state_space_size:], state_eval))
                        # Before we step in the environment, save the Mujoco state (qpos and qvel)
                        # episode_eval_dict['qpos'].append(env.sim.get_state()[1].tolist()) # qpos
                        # episode_eval_dict['qvel'].append(env.sim.get_state()[2].tolist()) # qvel
                        # Now we step forward in the environment by taking our action
                        action_noise_eval = np.random.normal(0, 0.1, action_space_size) if args.add_action_noise else 0
                        tmp_st_eval, reward_eval, done_eval, _ = env.step(action_eval + action_noise_eval)
                        next_state_eval = env.get_current_state(temp_state=tmp_st_eval, position_only=args.position_only)
                        # Add state noise if that parameter is true
                        next_state_eval += np.random.normal(0, 0.1, state_space_size) if args.add_state_noise else 0
                        # We have completed an evaluation step, now log it to the dictionary
                        # episode_eval_dict['state'].append(state_eval.tolist())
                        # episode_eval_dict['action'].append(action_eval.tolist())
                        if type(reward_eval) is float:
                            episode_eval_dict['reward'].append([reward_eval])
                        else:
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
                experiment.log_metric("Avg. Episode_Reward", avg_reward_eval,
                                      step=int(total_numsteps / args.eval_steps))

                # Log the episode reward to Comet.ml
                # Add the episode_dict
                eval_dicts.append(episode_eval_dict)
                experiment.log_asset_data(eval_dicts, name="output_metrics", step=int(total_numsteps / args.eval_steps), overwrite=True)
                # Log the model every 5 eval episodes
                if int(total_numsteps / args.eval_steps) % 5 == 0:
                    evl = str(int(total_numsteps / args.eval_steps))
                    act_path = "models/actor_eval_" + evl + ".model"
                    crt_path = "models/critic_eval_" + evl + ".model"
                    agent.save_model(args.env_name, actor_path=act_path, critic_path=crt_path)
                    experiment.log_asset(act_path)
                    experiment.log_asset(crt_path)
                # for item_str in episode_eval_dict.keys():
                #     item = episode_eval_dict[item_str]
                #     json_str = json.dumps(item, separators=(",", ":"), ensure_ascii=False).encode('utf8')
                #     item_name = 'episode_step_' + str(total_numsteps) + "_" + item_str
                #     experiment.log_asset_data(item, name=item_name, step=int(total_numsteps / args.eval_steps))

                # print("----------------------------------------")
                # print("Test Episodes: {}, Avg. Reward: {}".format(episodes_eval, round(avg_reward_eval, 2)))
                # print("----------------------------------------")

                # Now we are done evaluating. Before we leave, we have to set the state properly.
                env.set_state_after_eval(temp_state)

            if total_numsteps >= args.num_steps:
                stop_training = True
                break

            if PROFILING:
                print("Step time: ", time.time() - s_time)
        # Log to comet.ml
        experiment.log_metric("Episode_Reward", episode_reward, step=i_episode)
        std_log= np.mean(np.log(np.array(bstds)))
        experiment.log_metric("Base log stddev", std_log, step=i_episode)
        scale_log = np.mean(np.log(np.array(ascales)))
        experiment.log_metric("AR log scale", scale_log, step=i_episode)
        mean_critic_1_loss = np.mean(np.array(critic_1_losses))
        experiment.log_metric("Mean Critic 1 loss", mean_critic_1_loss, step=i_episode)
        mean_critic_2_loss = np.mean(np.array(critic_2_losses))
        experiment.log_metric("Mean Critic 2 loss", mean_critic_2_loss, step=i_episode)
        mean_policy_loss = np.mean(np.array(policy_losses))
        experiment.log_metric("Mean Policy Loss", mean_policy_loss)
        # Plot the entropy as well
        experiment.log_metric("Entropy parameter", alpha)

        if stop_training:
            break

# Save the final model before finishing program
agent.save_model(args.env_name,
                 actor_path="models/actor.model",
                 critic_path="models/critic.model")

# Log the models to comet in case we want to use them later.
experiment.log_asset("models/actor.model")
experiment.log_asset("models/critic.model")

env.close()
