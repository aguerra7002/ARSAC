# import this before torch
from comet_ml import Experiment

import argparse
import gym
import numpy as np
import itertools
import torch
import json
from arrl import ARRL
from tensorboardX import SummaryWriter
from replay_buffer import ReplayBuffer


parser = argparse.ArgumentParser(description='PyTorch AutoRegressiveFlows-RL Args')
# Once we get Mujoco then we will use this one
parser.add_argument('--env-name', default="HalfCheetah-v2",
                     help='Mujoco Gym environment (default: HalfCheetah-v2)')
#parser.add_argument('--env-name', default="LunarLanderContinuous-v2",
 #                   help='Box2D Environment (default: LunarLanderContinuous-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--action_lookback', type=int, default=5, metavar='G',
                    help='Use phi network to de-correlate time dependence and state by using previous action(s)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
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
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
#env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
action_space_size = env.action_space.sample().shape[0]
# Agent
print("Action Space", env.action_space)
agent = ARRL(env.observation_space.shape[0], env.action_space, args)

#TensorboardX
# writer = SummaryWriter(logdir='runs/{}_ARRL_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Comet logging
experiment = Experiment(api_key="tHDbEydFQGW7F1MWmIKlEvrly",
                        project_name="arsac", workspace="aguerra")
experiment.log_parameters(args.__dict__)

# Memory
memory = ReplayBuffer(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

with experiment.train():
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        lookback = args.action_lookback
        # Reset the previous action, as our program factors this into account when taking future actions
        prev_actions = np.zeros(action_space_size * lookback)

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state, prev_actions)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    # Log to TensorboardX
                    # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    # writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    # Log to Comet.ml
                    experiment.log_metric("Critic_1_Loss", critic_1_loss, step=updates)
                    experiment.log_metric("Critic_2_Loss", critic_2_loss, step=updates)
                    experiment.log_metric("Policy_Loss", policy_loss, step=updates)
                    experiment.log_metric("Entropy_Loss", ent_loss, step=updates)

                    updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            #print("Training: ", reward, "Action:", action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(prev_actions, state, action, reward, next_state, mask) # Append transition to memory
            state = next_state
            prev_actions = np.concatenate((prev_actions[:-action_space_size], action))

        if total_numsteps > args.num_steps:
            break
        # Log to tensorboardX
        # writer.add_scalar('reward/train', episode_reward, i_episode)
        # Log to comet.ml (needed for viewing results remotely
        experiment.log_metric("Epsiode_Reward", episode_reward, step=i_episode)
        # Log to console
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 15 == 0 and args.eval == True:
            agent.save_model(args.env_name,
                              actor_path="models/" + args.env_name + "_actor_" + str(lookback) + ".model",
                              critic_path="models/" + args.env_name + "_critic_" + str(lookback) + ".model")
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state_eval = env.reset()
                prev_actions_eval = np.zeros(action_space_size * lookback)
                episode_reward = 0
                done = False
                while not done:

                    action_eval = agent.select_action(state_eval, prev_actions_eval, eval=True)  # Sample action from policy

                    next_state_eval, reward, done, _ = env.step(action_eval)
                    if done:
                        break
                    episode_reward += reward
                    # print("Testing: ", reward, "Action:", action_eval)
                    state_eval = next_state_eval
                    prev_actions_eval = np.concatenate((prev_actions_eval[:-action_space_size], action_eval))
                avg_reward += episode_reward
            avg_reward /= episodes

            # Log to TensorboardX
            # writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            # Log to Comet.ml
            experiment.log_metric("Avg_Episode_Reward", avg_reward, step=i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

# Save the final model before finishing program
agent.save_model(args.env_name,
                         actor_path="models/" + args.env_name + "_actor_" + str(lookback) + ".model",
                         critic_path="models/" + args.env_name + "_critic_" + str(lookback) + ".model")

# Now we view the model perform before closing. NOTE: You should use view_model_performance.py to see model performance.
# state = env.reset()
# prev_action = torch.zeros(env.action_space.sample().shape)
# for _ in range(1000):
#     env.render()
#     action = agent.select_action(state, prev_action, eval=True)
#     next_state, reward, done, _ = env.step(action)
#     prev_action = action
#     state = next_state
# time.sleep(2)
env.close()