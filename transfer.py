
from comet_ml.api import API
from comet_ml import Experiment
import os
import argparse
import numpy as np
import itertools
import json
import torch
from arrl import ARRL
from replay_buffer import ReplayBuffer
from pixelstate import PixelState
from env_wrapper import EnvWrapper
import time

# comet
api_key = 'tHDbEydFQGW7F1MWmIKlEvrly'

workspace = 'aguerra'
project_name = 'arsac-test'
comet_api = API(api_key=api_key)

# PUT THE NAME OF THE FILE WE WANT TO SAVE HERE
actor_filename = "actor.model"
critic_filename = "critic.model"

parser = argparse.ArgumentParser(description='PyTorch AutoRegressiveFlows-RL Args')
# Once we get Mujoco then we will use this one
parser.add_argument('--experiment_id', default="14ae9513a53a42bf89b65d02f6cdc5e7",
                    help='Experiment ID we want to transfer our experiment from')
parser.add_argument('--task-name', default="walk",
                    help='Transfer task')
parser.add_argument('--transfer_flow', default=True,
                    help="Determines whether or not we transfer the flow network")
parser.add_argument('--transfer_base', default=False,
                    help="Determines whether or not we transfer the base network")
parser.add_argument('--transfer_critic', default=False,
                    help="Determines whether or not we transfer the critic network")
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=3000000, metavar='N',
                    help='maximum number of steps (default: 3000000)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Number of steps we take to fill the replay buffer.')
parser.add_argument('--freeze_steps', type=int, default=20000, metavar='N',
                    help='number of steps we run without updating the flow network')
parser.add_argument('--rbo_increase_factor', type=float, default=1.0, metavar='N',
                    help='determines how much we increase the restrict_base_output parameter after each episode.')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')
parser.add_argument('--device_id', type=int, default=0, metavar='G',
                    help='Which GPU to run on')
args = parser.parse_args()
# Adjust these to determine what component of the policy gets transferred.

base_experiment = comet_api.get_experiment(project_name=project_name,
                                                  workspace=workspace,
                                                  experiment=args.experiment_id)
asset_list = base_experiment.get_asset_list()

# First setup the arguments
args_asset_id = [x for x in asset_list if x['fileName'] == "args"][0]['assetId']
args_dict = base_experiment.get_asset(args_asset_id, return_type="json")
for key in args_dict.keys():
    if key not in args.__dict__.keys():
        args.__dict__[key] = args_dict[key]

# Setup Cuda
if args.device_id is None:
    args.cuda = False
else:
    args.cuda = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.cuda.set_device(0)

# Next we setup the environment
transfer_domain = args.env_name
transfer_task = args.task_name
env = EnvWrapper(transfer_domain, transfer_task, args.pixel_based, args.resolution)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
# Action Space Size
action_space_size = env.action_space.sample().shape[0]

# Figure out what the state size is
state_space_size = env.get_state_space_size(position_only=args.position_only)

# Now initialize the agent
agent = ARRL(state_space_size, env.action_space, args)

# Here is the transfer component. For now, we only transfer the actor.
actor_asset_id = [x for x in asset_list if actor_filename == x['fileName']][0]['assetId']
actor = base_experiment.get_asset(actor_asset_id)
act_path = 'tmploaded/actor.model'
with open(act_path, 'wb+') as f:
    f.write(actor)

if args.transfer_critic:
    crt_path = 'tmploaded/critic.model'
    critic_asset_id = [x for x in asset_list if critic_filename == x['fileName']][0]['assetId']
    critic = base_experiment.get_asset(critic_asset_id)
    with open(crt_path, 'wb+') as f:
        f.write(critic)
    agent.load_model(act_path, crt_path, flow_only=args.transfer_flow, base_only=args.transfer_base)
else:
    agent.load_model(act_path, None, flow_only=args.transfer_flow, base_only=args.transfer_base)

if args.freeze_steps > 0:
    agent.require_flow_grad(False) # This will freeze the flow network

# Comet logging. Note we are starting a new experiment now
experiment = Experiment(api_key=api_key,
                        project_name=project_name, workspace=workspace)
experiment.log_parameters(args.__dict__)
json_str = json.dumps(args.__dict__)
experiment.log_asset_data(json_str, name="args")

# This will be what we use to get image states for the main loop. The updating parameter uses a multithreaded version of this class
if args.pixel_based:
    state_getter_main = PixelState(1, args.env_name, args.resolution, state_space_size)

# Will hold ALL of our results and will be what we log to comet
eval_dicts = []

# Training Loop
total_numsteps = 0
updates = 0
stop_training = False

action_lookback = max(args.action_lookback_actor, args.action_lookback_critic)
state_lookback = max(args.state_lookback_actor, args.state_lookback_critic)

# Memory
memory = ReplayBuffer(args.replay_size)
rbo = args.restrict_base_output
print("RBO", rbo)
with experiment.train():
    for i_episode in itertools.count(1):

        episode_reward = 0
        episode_steps = 0
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
            # print("Step #:", total_numsteps)
            # prev_time = time.time() # TODO: Remove later
            # s_time = prev_time

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
                    action = agent.select_action(state_n, prev_states_n, prev_actions, random_base=args.random_base_train)
                else:
                    action = agent.select_action(state, prev_states, prev_actions, random_base=args.random_base_train)

            # print("select action:", time.time() - prev_time) # TODO: Remove later
            # prev_time = time.time()

            # Unfreeze the flow network if we are past the number of freeze steps
            if total_numsteps == args.start_steps + args.freeze_steps + 1:
                agent.require_flow_grad(True)

            if len(memory) > args.start_steps: #args.batch_size * :
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    # print("update params:")  # TODO: Remove later
                    # What we had before

                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates, restrict_base_output=rbo)
                    # print("Entropy Parameter", alpha)
                    # Log to Comet.ml
                    # experiment.log_metric("Critic_1_Loss", critic_1_loss, step=updates)
                    # experiment.log_metric("Critic_2_Loss", critic_2_loss, step=updates)
                    # experiment.log_metric("Policy_Loss", policy_loss, step=updates)
                    # experiment.log_metric("Entropy_Loss", ent_loss, step=updates)
                    # Put it on a different thread
                    #threads.append(threading.Thread(target=agent.update_parameters, args=(memory, args.batch_size, updates)))
                    # Start the new thread
                    #threads[-1].start()

                    updates += 1

            prev_time = time.time() # TODO: REmove later

            action_noise = np.random.normal(0, 0.1, action_space_size) if args.add_action_noise else 0
            # Take a step in the environment. Note, we get the next state in the following line in case we only want pos
            tmp_st, reward, done, _ = env.step(action + action_noise)  # Step
            next_state = env.get_current_state(temp_state=tmp_st, position_only=args.position_only)

            # print("get state:", time.time() - prev_time)  # TODO: Remove later
            # prev_time = time.time()

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

            # print("other stuff:", time.time() - prev_time)  # TODO: Remove later
            # prev_time = time.time()

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

            if total_numsteps >= args.num_steps + args.start_steps:
                stop_training = True
                break
            # print("Step time: ", time.time() - s_time)

        # Here is where we increase the rbo factor.
        if total_numsteps >= args.start_steps:
            rbo *= args.rbo_increase_factor
        # Log to comet.ml
        experiment.log_metric("Episode_Reward", episode_reward, step=i_episode)
        # Log to console
        # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
        #                                                                              episode_steps,
        #                                                                              round(episode_reward, 2)))

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
